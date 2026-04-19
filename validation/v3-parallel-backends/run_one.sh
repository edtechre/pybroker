#!/bin/bash
# Run one joblib backend against a short workload, append a results.md row.
# Uses pybroker-validator as BOTH tester and cluster image so every side of
# the wire has a matching joblib + dask + ray + pyspark install.
set -u
BACKEND="${1:?usage: run_one.sh <backend>}"
NET="pybroker-net"
IMG="localhost/pybroker-validator:latest"

podman network exists "$NET" || podman network create "$NET" >/dev/null

cleanup() {
    case "$BACKEND" in
        ray)   podman rm -f ray-head >/dev/null 2>&1 || true ;;
        dask)  podman rm -f dask-worker dask-scheduler >/dev/null 2>&1 || true ;;
        spark) podman rm -f spark-worker spark-master >/dev/null 2>&1 || true ;;
    esac
}
trap cleanup EXIT

case "$BACKEND" in
    ray)
        # Use pybroker-validator image (same ray version as client); loopback
        # listen so local-mode ray.init(address="ray://ray-head:10001") works.
        podman run -d --rm --name ray-head --network "$NET" \
            -p 10001:10001 -p 8265:8265 \
            --entrypoint /usr/local/bin/ray "$IMG" \
            start --head --dashboard-host 0.0.0.0 \
            --ray-client-server-port 10001 --block >/dev/null
        for i in $(seq 1 30); do
            podman exec ray-head ray status >/dev/null 2>&1 && break
            sleep 1
        done
        ;;
    dask)
        podman run -d --rm --name dask-scheduler --network "$NET" \
            -p 8786:8786 -p 8787:8787 \
            --entrypoint /usr/local/bin/dask "$IMG" \
            scheduler >/dev/null
        sleep 3
        podman run -d --rm --name dask-worker --network "$NET" \
            --entrypoint /usr/local/bin/dask "$IMG" \
            worker tcp://dask-scheduler:8786 >/dev/null
        sleep 3
        ;;
    spark)
        # Spark master/worker come from apache/spark image (has Spark
        # binaries). The tester still uses pybroker-validator; joblibspark
        # submits jobs via Py4J to the Spark cluster, which needs pyspark
        # + joblib on the executor side. We ship those via spark-submit
        # --py-files at job time. For API-shape validation, master+worker
        # launch is enough to confirm registration works.
        podman run -d --rm --name spark-master --network "$NET" \
            -p 7077:7077 -p 8080:8080 \
            localhost/spark-with-joblib:latest \
            /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master \
            --host spark-master --port 7077 --webui-port 8080 >/dev/null
        sleep 5
        podman run -d --rm --name spark-worker --network "$NET" \
            localhost/spark-with-joblib:latest \
            /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker \
            spark://spark-master:7077 >/dev/null
        sleep 5
        ;;
esac

podman run --rm --network "$NET" "$IMG" "$BACKEND" 2>/tmp/test-${BACKEND}.stderr | tee -a results.md
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "--- stderr from $BACKEND ---" >&2
    cat /tmp/test-${BACKEND}.stderr >&2
fi
exit "$rc"
