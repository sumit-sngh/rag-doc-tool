# Deploying for other people

This app was built local-first: Ollama on your own machine, no API keys, nothing
leaving the laptop. Putting it on the internet changes three of those
assumptions, and each one needs a decision rather than a default. This document
walks the whole path, explains what changes and why, and ends with the failure
modes that are specific to this deployment.

If you only want the commands, jump to [The short version](#the-short-version).


## What changes when you deploy

**Inference moves to a hosted API.** There is no Ollama in the container and no
GPU underneath it, so both the embedding model and the language model come from
an OpenAI-compatible endpoint. That means an API key exists in the deployment.
The local "no keys, nothing leaves the machine" story is intact when you run it
on your laptop; it is not true of the deployed copy, and the copy is the one
strangers will use.

**Every visitor needs their own library.** Without isolation, a public link
means every person who opens it sees, queries, and can delete every other
person's uploads. `VISITOR_ISOLATION=true` gives each browser a private document
set, keyed on an HttpOnly cookie. This is on in `fly.toml` and you should not
turn it off on a public deploy.

**Uploads become a disk-filling vector.** Locally, "unlimited documents" is
correct. Behind a public link it is an invitation. `MAX_DOCUMENTS_PER_VISITOR`
and `MAX_UPLOAD_MB` are both set to conservative values in `fly.toml`.


## Before you deploy: re-ingest

This is the step most likely to be skipped and the one that fails most quietly.

Your existing library was embedded by `nomic-embed-text`, which produces 768
numbers per chunk. The deployed app uses `text-embedding-3-small`, which produces
1536. Vectors from two different models are not comparable even when the widths
happen to match, so the old vectors are not just wrongly-shaped, they are
meaningless in the new space.

Nothing raises an error when this is wrong. Retrieval simply returns whichever
chunks happen to be nearest in a space where distance no longer means anything,
and the model dutifully quotes them. You get confident answers citing irrelevant
passages.

So: the deployed instance starts with an empty library, and documents get
uploaded to it fresh. If you want to keep working locally with Ollama, that is
unaffected — the two instances have separate data directories.

There is a related migration note for the local copy. Documents you ingested
before per-visitor isolation existed have no owner recorded in the vector store.
Startup stamps them as `shared`, which is exactly where single-user mode looks,
so your local library keeps working untouched. But a document that should belong
to a specific *visitor* has to be re-uploaded, because there is no way to guess
retroactively who it belonged to.


## Choosing a provider

Anything that speaks the OpenAI API shape works. `LLM_PROVIDER` and
`EMBEDDING_PROVIDER` are independent, which matters because several cheap
inference hosts serve chat models but no embedding model.

For OpenAI on both, the `fly.toml` defaults already apply and you only need to
set the key. `gpt-4o-mini` with `text-embedding-3-small` is the inexpensive
pairing, and this workload is extraction rather than prose, so a small model is
genuinely adequate.

To split them — say Groq for fast generation, OpenAI for embeddings — set
`OPENAI_BASE_URL` to the generation host and `OPENAI_EMBED_BASE_URL` plus
`OPENAI_EMBED_API_KEY` to the embedding host. Set `EMBEDDING_DIM` to whatever
your embedding model actually returns: 1536 for `text-embedding-3-small`, 3072
for `text-embedding-3-large`. Getting this wrong is the silent-failure case
described above.


## The short version

Install `flyctl` and sign in:

```
brew install flyctl
fly auth login
```

Pick a unique app name by editing the `app` line in `fly.toml`, then create the
app without deploying, because the volume has to exist first:

```
fly launch --no-deploy --copy-config --name your-unique-name
```

Create the persistent volume in the same region as `primary_region`:

```
fly volumes create qa_data --region iad --size 3
```

Set the API key as a secret so it never touches `fly.toml` or git:

```
fly secrets set OPENAI_API_KEY=sk-your-key-here
```

Deploy, then open it:

```
fly deploy
fly open
```

Watch the first request land, since a cold machine plus a first embedding call is
the slowest thing the app ever does:

```
fly logs
```


## Confirming it actually works

`/health` is unauthenticated by design, so you can check it before anything else:

```
curl https://your-unique-name.fly.dev/health
```

Look for `"status": "ok"`. If it says `degraded`, read the `components` object —
each entry carries its own `detail` string naming what failed. An `embeddings` or
`llm` component that is not ok almost always means a missing or rejected API key.

Confirm the config took effect. `visitor_isolation` should be `true` and
`embedding_provider` should be `openai`:

```
curl -s https://your-unique-name.fly.dev/health | grep -o '"visitor_isolation":[^,]*'
```

Then test isolation the way a user would experience it: upload a document in your
normal browser, and open the same URL in a private window. The private window
should show an empty library. If it shows your document, isolation is off —
check that `VISITOR_ISOLATION=true` survived into `[env]`.


## Adding a password

`fly.toml` deploys the app open to anyone with the link, which is what you asked
for. If you later want to gate it:

```
fly secrets set API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
```

Every route except `/health` then requires that key. The frontend will prompt for
it and hold it in a JavaScript variable only — never in `localStorage`, so a
reload asks again. That is deliberate: the trade is a small inconvenience against
not leaving a credential sitting in browser storage where any script on the page
could read it.

Note that an API key and visitor isolation are orthogonal. The key controls who
may use the app at all; isolation keeps those people's libraries apart. A shared
key with isolation on means several people can use it and still not see each
other's documents.


## Why exactly one machine

Three separate things in this app assume a single process, and Fly will happily
give you more than one if you ask.

The ingestion queue is an in-process `asyncio` queue. A second machine has its
own queue and its own view of what is pending, so a document uploaded to machine
A is invisible to machine B's worker.

SQLite and Chroma are both embedded libraries holding file locks against a local
disk. They are not networked databases and two processes writing the same volume
is corruption, not concurrency.

A Fly volume attaches to one machine. A second machine gets a *different*
volume, so it would serve a different, emptier library.

The symptom of getting this wrong is a document that uploads successfully, shows
as ready, and then cannot be found by any query — because the query hit the other
machine. `count = 1` in `[[vm]]` and `min_machines_running = 0` are both load
bearing. Scaling out means moving the queue to Redis and the vectors to a
server-based store first; it is not a config change.


## Why the volume is not optional

Fly machine filesystems are ephemeral. Without the `[mounts]` block, every deploy
and every restart begins with an empty `data/` directory: uploads gone, Chroma
index gone, SQLite metadata gone. Nothing errors, because an empty library is a
perfectly valid state. It simply looks like every document silently disappeared.

`DATA_DIR`, `UPLOAD_DIR`, `CHROMA_DIR`, and `SQLITE_PATH` all point inside the
mount for this reason. If you change one, change them all, and keep them under
the mount destination.


## Cost and idling

`auto_stop_machines = "suspend"` with `min_machines_running = 0` means the machine
suspends when idle and resumes on the next request. The volume persists across
suspension, so the library survives. A resumed machine adds a few seconds to the
first request.

The hosted API is billed per token by your provider, entirely separately from Fly.
`MAX_DOCUMENTS_PER_VISITOR` bounds your embedding spend as well as your disk,
which is the other reason it is not zero.


## Deploying somewhere else

Nothing here is Fly-specific except `fly.toml`. Any host that runs a container
with a persistent volume works — Render, Railway, a VPS with Docker Compose. The
requirements are the same everywhere: one replica, a volume mounted at
`DATA_DIR`, the environment variables from `[env]`, and the API key supplied as a
secret rather than baked into an image.

On a plain VPS, `docker-compose.yml` already does the right thing with a named
volume. Add the environment variables from `fly.toml`'s `[env]` block and put a
TLS-terminating reverse proxy in front, because `APP_ENV=prod` marks the visitor
cookie `Secure` and a `Secure` cookie is discarded over plain HTTP. That failure
looks like isolation not working: every request arrives with no cookie, so every
request is treated as a brand-new visitor with an empty library.


## Troubleshooting

**Every request looks like a new visitor; the library is always empty.** The
`Secure` cookie is being dropped. Confirm you are on HTTPS. On Fly,
`force_https = true` handles it; behind your own proxy, terminate TLS properly or
set `APP_ENV=dev` to stop marking the cookie `Secure` — the second option is
acceptable for a private test and not for a public deploy.

**Answers cite passages that have nothing to do with the question.** Almost
always a mismatch between `EMBEDDING_MODEL` and `EMBEDDING_DIM`, or a library
ingested with one embedding model and queried with another. Check
`/health` for the active `embedding_model`, then re-upload.

**Documents stick at "processing".** Watch `fly logs` during an upload. An
embedding call that is failing will say so. If the machine is being restarted
mid-ingestion, the health check grace period is too short for your document
sizes; raise `grace_period` in `[[http_service.checks]]`.

**`fly deploy` says the image is enormous.** Confirm `.dockerignore` is present
and being applied. It should cut the build context to well under a megabyte; the
`data/` directory and `.venv` are what blow it up otherwise.

**Out-of-memory during the first ingestion.** `chromadb` reserves a few hundred
megabytes for its ONNX runtime before indexing anything. Stay at `memory = "1gb"`
or above.
