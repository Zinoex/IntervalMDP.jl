const path = require('path');
const sqlite3 = require('sqlite3').verbose();

const DB_PATH = path.join(__dirname, 'telemetry.db');

let dbInstance = null;
function getDb() {
  if (dbInstance) return dbInstance;
  dbInstance = new sqlite3.Database(DB_PATH);
  dbInstance.serialize(() => {
    dbInstance.run(
      `CREATE TABLE IF NOT EXISTS events (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         ts TEXT NOT NULL DEFAULT (datetime('now')),
         event_name TEXT NOT NULL,
         details TEXT
       )`
    );
  });
  return dbInstance;
}

function recordEvent(eventName, details = {}) {
  const db = getDb();
  const payload = typeof details === 'string' ? details : JSON.stringify(details);
  db.run('INSERT INTO events (event_name, details) VALUES (?, ?)', [eventName, payload]);
}

module.exports = { recordEvent, getDb, DB_PATH };

if (require.main === module) {
  const [, , cmd, name, detailsJson] = process.argv;
  if (cmd === 'recordEvent') {
    let details = {};
    if (detailsJson) {
      try { details = JSON.parse(detailsJson); } catch { details = { raw: detailsJson }; }
    }
    recordEvent(name, details);
    getDb().close(() => console.log(`recorded: ${name}`));
  } else {
    console.error('usage: telemetry.js recordEvent <name> [jsonDetails]');
    process.exit(1);
  }
}
