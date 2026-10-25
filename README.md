# Apple Watch Lab

## PulsHealth sync (protocol v1)

The FastAPI service accepts PulsHealth uploads at `POST /v1/batches`. It uses the existing
`WEBHOOK_TOKEN` as a bearer token and pins uploads to one user ID (`PULS_USER_ID`). If that
variable is unset, the receiver uses PulsHealth's default user ID:
`5ea4d000-0000-4000-8000-000000000001`.

### Configure Render

1. Deploy the project with `puls_receiver.py` and the updated `main.py` together.
2. Keep `MONGO_URI` and `WEBHOOK_TOKEN` set in Render. Optionally set `PULS_USER_ID` to the UUID
   you will enter in PulsHealth. Do not put the bearer token in a URL or commit it to this repo.
3. Make sure the Render service is reachable over HTTPS. The app's URL should be the service base
   URL, for example `https://your-service.onrender.com`; do not append `/v1/batches`.

### Pair PulsHealth

In PulsHealth, open **Sync → Database → Set Up** (or **Database** if already configured), enter the
Render base URL, the exact `WEBHOOK_TOKEN` as the bearer token, and the same `PULS_USER_ID`. Save
and apply the connection, grant read access only to the data types you want, then run **Sync Now**.
The free plan currently allows five types; ECG is Apple Watch-only and is retained with its voltage
trace. HealthKit background delivery is controlled by iOS, so open the app if you need it to catch up.

### What the receiver stores

- It accepts bounded gzip/NDJSON batches, validates the complete batch before writing, and returns
  success only after MongoDB writes finish. Retry-safe unique keys prevent duplicate samples.
- Samples without explicit Apple Watch source/device evidence are excluded. ECGs are Watch-only by
  definition; aggregates are retained only when their device filter is `watch`; workout routes and
  streams are retained only when their parent workout is confirmed as Watch-sourced.
- Supported quantity samples also refresh the site's existing five-minute dashboard bins. Sleep
  stages and ECGs feed the existing sleep and ECG views. Full PulsHealth samples are kept in
  `puls_samples`; aggregates, activity rings, workout routes/streams, profiles, and receipts use
  separate `puls_*` collections.
- The endpoint is single-user by design. `X-User-ID` must match `PULS_USER_ID`; the token is shared
  with the existing webhook. The app's data stays in your configured MongoDB. Render's local disk is
  used only for temporary request files, which are removed after each request.

Check `/stats?token=...` for MongoDB storage and PulsHealth sample/batch counts. MongoDB Atlas free
clusters have a 512 MB storage cap including indexes, so monitor that number. Do not upload actual
health data to an unverified receiver; use a synthetic batch to verify the deployment first.
