# Acme Search API: Rate Limits

This is a fictional internal document used in the docrag demo to show ingesting a new file.

## Limits per plan

Every API key is limited per minute, counted over a sliding 60-second window.

- **Free plan:** 60 requests per minute to `/v2/search`, and 10 requests per minute to
  `/v2/export`.
- **Team plan:** 600 requests per minute to `/v2/search`, and 100 requests per minute to
  `/v2/export`.

Bulk exports larger than 10,000 rows count as 5 requests against the `/v2/export` limit.

## When a limit is exceeded

The API responds with HTTP status `429 Too Many Requests` and a `Retry-After` header that gives
the number of seconds to wait. Clients should wait at least that long before retrying, and
should add random jitter of up to one second so that many clients don't retry at the same
moment.

Keys that exceed their limit more than 1,000 times in one day are suspended for 24 hours.
