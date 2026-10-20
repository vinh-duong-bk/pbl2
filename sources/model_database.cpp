#include "model_database.h"

#include <sqlite3.h>

namespace {
bool execute(sqlite3 *db, const char *sql) {
  char *error = nullptr;
  const int result = sqlite3_exec(db, sql, nullptr, nullptr, &error);
  sqlite3_free(error);
  return result == SQLITE_OK;
}
}

bool recordModelMetadata(const std::string &databasePath,
                         const std::string &modelPath,
                         int imageSize,
                         int boostAmount,
                         int learnerCount,
                         double trainingError) {
  sqlite3 *db = nullptr;
  if (sqlite3_open(databasePath.c_str(), &db) != SQLITE_OK) {
    sqlite3_close(db);
    return false;
  }

  const char *schema =
      "CREATE TABLE IF NOT EXISTS models ("
      "id INTEGER PRIMARY KEY AUTOINCREMENT,"
      "model_path TEXT NOT NULL,"
      "image_size INTEGER NOT NULL,"
      "boost_amount INTEGER NOT NULL,"
      "learner_count INTEGER NOT NULL,"
      "training_error REAL NOT NULL,"
      "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
      ");";
  if (!execute(db, schema)) {
    sqlite3_close(db);
    return false;
  }

  sqlite3_stmt *statement = nullptr;
  const char *insert =
      "INSERT INTO models "
      "(model_path, image_size, boost_amount, learner_count, training_error) "
      "VALUES (?, ?, ?, ?, ?);";
  if (sqlite3_prepare_v2(db, insert, -1, &statement, nullptr) != SQLITE_OK) {
    sqlite3_close(db);
    return false;
  }
  sqlite3_bind_text(statement, 1, modelPath.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(statement, 2, imageSize);
  sqlite3_bind_int(statement, 3, boostAmount);
  sqlite3_bind_int(statement, 4, learnerCount);
  sqlite3_bind_double(statement, 5, trainingError);
  const bool success = sqlite3_step(statement) == SQLITE_DONE;
  sqlite3_finalize(statement);
  sqlite3_close(db);
  return success;
}
