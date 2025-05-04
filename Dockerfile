FROM rust:1-bookworm as BUILDER

RUN sed -i 's/main/main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt update \
    && apt install -y libopencv-dev intel-mkl-full clang cmake

WORKDIR /app

COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN sed -i 's/main/main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt update \
    && apt install -y intel-mkl-full libopencv-core406 libopencv-highgui406 libopencv-flann406 libopencv-imgproc406 libopencv-contrib406 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=BUILDER /app/target/release/imsearch /usr/local/bin/imsearch

ENTRYPOINT ["imsearch"]