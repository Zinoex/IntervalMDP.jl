# Hello World API — Specification

## Objective
Deliver a minimal Node.js HTTP API exposing a single `GET /hello` endpoint that returns a JSON greeting, with Jest + supertest tests covering success cases and unknown-route handling.

## Stack & Dependencies
- **Runtime:** Node.js (>= 18)
- **Framework:** Express (^4.19.x)
- **Test runner:** Jest (^29.x)
- **HTTP assertions:** supertest (^7.x)

### package.json
- `name`: `hello-world-api`
- `version`: `1.0.0`
- `main`: `src/server.js`
- `scripts`:
  - `start`: `node src/server.js`
  - `test`: `jest`
- `dependencies`: `express`
- `devDependencies`: `jest`, `supertest`

## Project Layout
```
hello-world-api/
├── package.json
├── .gitignore                # node_modules, coverage
├── src/
│   ├── app.js                # Express app factory (no .listen) — exported for tests
│   └── server.js             # Imports app, calls app.listen(PORT)
└── tests/
    └── hello.test.js         # Jest + supertest suite
```

## Module Contracts

### `src/app.js`
- Exports an Express `app` instance (module.exports = app).
- Registers `GET /hello` handler.
- Registers a fallback 404 handler for any unmatched route returning JSON `{"error":"Not Found"}` with status `404`.
- Does **not** call `app.listen` (keeps it import-safe for tests).

### `src/server.js`
- `const app = require('./app');`
- Reads `PORT` from `process.env.PORT`, defaults to `3000`.
- Calls `app.listen(PORT, () => console.log(...))`.

## Endpoint Specification

### `GET /hello`
- **Query params:**
  - `name` (optional, string)
- **Response:** `200 OK`, `Content-Type: application/json`
- **Body:**
  - No `name` (or empty string): `{"message":"Hello, World!"}`
  - With `name=Foo`: `{"message":"Hello, Foo!"}`
- Name value is interpolated as-is (no sanitization required for this scope).

### Any other route / method
- **Response:** `404 Not Found`, JSON body `{"error":"Not Found"}`.

## Test Specification (`tests/hello.test.js`)
Use `const request = require('supertest'); const app = require('../src/app');`.

Required test cases:
1. `GET /hello` → status `200`, body equals `{ message: 'Hello, World!' }`.
2. `GET /hello?name=Foo` → status `200`, body equals `{ message: 'Hello, Foo!' }`.
3. `GET /does-not-exist` → status `404`, body equals `{ error: 'Not Found' }`.

## Acceptance Criteria
- [ ] `npm install` completes without errors.
- [ ] `npm test` runs Jest and all three tests pass.
- [ ] `npm start` starts a server on port 3000 (or `$PORT`); `curl localhost:3000/hello` returns `{"message":"Hello, World!"}`.
- [ ] `curl "localhost:3000/hello?name=Foo"` returns `{"message":"Hello, Foo!"}`.
- [ ] `curl -i localhost:3000/unknown` returns HTTP 404 with JSON `{"error":"Not Found"}`.
- [ ] `app.js` does not bind a port (importable in tests without side effects).

## File List (to be created)
- `package.json`
- `.gitignore`
- `src/app.js`
- `src/server.js`
- `tests/hello.test.js`

## Out of Scope
- Authentication, logging middleware, input validation/sanitization, CORS, Dockerfile, CI config, TypeScript.
