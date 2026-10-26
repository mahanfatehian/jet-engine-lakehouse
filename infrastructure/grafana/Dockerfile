# =============================================================================
# Grafana OSS with the official ClickHouse datasource plugin baked in at build time.
#
# Why bake instead of GF_PLUGINS_PREINSTALL: a runtime install downloads the plugin on every
# start of a fresh container. Offline, behind a flaky proxy, or where grafana.com's plugin
# storage (storage.googleapis.com) refuses the connection, Grafana then starts without the
# plugin and every panel fails with "plugin not found". Baked in, the network is needed once,
# at build time (the layer is cached afterwards), every restart works offline, and the plugin
# changes only with this file.
#
# grafana/grafana is the OSS edition (Grafana's Docker docs name it so). The grafana/grafana-oss
# repository no longer receives updates: its newest tag is 13.0.2, while 13.0.x is at 13.0.11 and
# 13.2.4 is current. Pinned by tag and multi-arch index digest; any Grafana >= 11.6 satisfies
# the plugin (its grafanaDependency).
#
# Build behind a proxy, or from a mirror of the same archives (the digests below still apply):
#   docker compose -f infrastructure/docker-compose.yml build \
#     --build-arg HTTPS_PROXY=http://<proxy-host>:<port> grafana
#   ... --build-arg CLICKHOUSE_PLUGIN_URL='https://<mirror>/{version}/linux-{arch}.zip' grafana
# =============================================================================
ARG GRAFANA_IMAGE=grafana/grafana:13.2.4@sha256:dbbc38ea21103cefb50396340cd88afec779c7e070b0a7bf5a75b4ce46651d3f
FROM ${GRAFANA_IMAGE}

ARG TARGETARCH
ARG CLICKHOUSE_PLUGIN_VERSION=4.23.2
# SHA-256 of the linux archives of that version as published by grafana.com
# (GET https://grafana.com/api/plugins/grafana-clickhouse-datasource/versions/4.23.2).
ARG CLICKHOUSE_PLUGIN_SHA256_AMD64=6a746f4d49a9eb01bfa97986407b2eea6b3ca3c38ecb262a3ffa26febc2ee450
ARG CLICKHOUSE_PLUGIN_SHA256_ARM64=95ed23fc6f0232e275cf0db0bca494277de78331922a82e49751ae8bbcbcf2e7
# {version} and {arch} (amd64 | arm64) are substituted before the download.
ARG CLICKHOUSE_PLUGIN_URL=https://grafana.com/api/plugins/grafana-clickhouse-datasource/versions/{version}/download?os=linux&arch={arch}

# The plugin lives in the image, outside the /var/lib/grafana data volume, so a volume created
# by an older image can never shadow it with a stale copy. The directory is root-owned and
# read-only for Grafana (uid 472): plugins change by rebuilding, never from the UI. Grafana 13
# would otherwise download ~20 default plugins (drilldown apps, decoupled core datasources) in
# the background on every start; none is used here, so that preinstall is off, like usage
# reporting and update checks: an offline restart makes no network calls at all.
ENV GF_PATHS_PLUGINS=/usr/share/grafana/plugins-baked \
    GF_PLUGINS_PREINSTALL_DISABLED=true \
    GF_PLUGINS_PLUGIN_ADMIN_ENABLED=false \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false \
    GF_NEWS_NEWS_FEED_ENABLED=false \
    GF_SECURITY_DISABLE_GRAVATAR=true

USER root
# Resumable download with bounded retries (curl inside RUN honours the build proxy), SHA-256
# check against the pinned digest, then unpack. Grafana additionally verifies the plugin's
# Grafana Labs signature (MANIFEST.txt) every time it loads it.
RUN set -eu; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    case "$arch" in \
        amd64|x86_64) arch=amd64; digest="$CLICKHOUSE_PLUGIN_SHA256_AMD64" ;; \
        arm64|aarch64) arch=arm64; digest="$CLICKHOUSE_PLUGIN_SHA256_ARM64" ;; \
        *) echo "no pinned ClickHouse plugin digest for architecture $arch" >&2; exit 1 ;; \
    esac; \
    url="$(printf '%s' "$CLICKHOUSE_PLUGIN_URL" \
        | sed -e "s/{version}/$CLICKHOUSE_PLUGIN_VERSION/g" -e "s/{arch}/$arch/g")"; \
    archive=/tmp/grafana-clickhouse-datasource.zip; \
    attempt=1; \
    until curl -fsSL --connect-timeout 20 --speed-limit 1024 --speed-time 60 -C - -o "$archive" "$url"; do \
        if [ "$attempt" -ge 6 ]; then \
            echo "plugin download failed: $url (set HTTPS_PROXY or CLICKHOUSE_PLUGIN_URL)" >&2; \
            exit 1; \
        fi; \
        sleep $((attempt * 5)); \
        attempt=$((attempt + 1)); \
    done; \
    echo "$digest  $archive" | sha256sum -c -; \
    mkdir -p "$GF_PATHS_PLUGINS"; \
    unzip -q "$archive" -d "$GF_PATHS_PLUGINS"; \
    rm -f "$archive"; \
    plugin_dir="$GF_PATHS_PLUGINS/grafana-clickhouse-datasource"; \
    grep -q "\"version\": \"$CLICKHOUSE_PLUGIN_VERSION\"" "$plugin_dir/plugin.json"; \
    chmod -R a+rX,go-w "$GF_PATHS_PLUGINS"; \
    chmod 0755 "$plugin_dir/gpx_clickhouse_linux_$arch"
USER 472
