CREATE TABLE "image" (
    -- 图片 ID
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- 图片 blake3 哈希
    hash BLOB UNIQUE NOT NULL,
    -- 图片路径
    path TEXT NOT NULL
);

CREATE TABLE "vector_stats" (
    -- 图片 ID
    id INTEGER PRIMARY KEY,
    -- 特征向量数量
    vector_count INTEGER NOT NULL,
    -- 截止到目前位置的特征向量总数
    total_vector_count INTEGER NOT NULL,
    -- 特征向量大小
    indexed BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE "vector" (
    -- 图片 ID
    id INTEGER PRIMARY KEY,
    -- 多维向量，维数为 count * 512bit
    vector BLOB NOT NULL
);

CREATE INDEX "idx_vector_stats_total_vector_count" ON "vector_stats" ("total_vector_count");