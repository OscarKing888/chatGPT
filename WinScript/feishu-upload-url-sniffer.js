(() => {
  const records = [];
  const redact = (url) =>
    String(url).replace(/([?&](?:token|access_token|tenant_access_token|authorization|X-Amz-Signature|x-amz-signature|sign|signature|expire|expires|auth_key|x-oss-signature)=)[^&]+/gi, '$1<redacted>');
  const add = (kind, method, url, detail) => {
    try {
      const u = new URL(url, location.href);
      const row = {
        at: new Date().toISOString(),
        kind,
        method: method || 'GET',
        host: u.host,
        url: redact(u.href),
        detail: detail || ''
      };
      records.push(row);
      console.log('[feishu-url-sniffer]', row.method, row.host, row.url, row.detail);
    } catch (_) {
      console.log('[feishu-url-sniffer]', kind, method, url, detail || '');
    }
  };

  const originalFetch = window.fetch;
  window.fetch = function patchedFetch(input, init = {}) {
    const url = typeof input === 'string' ? input : input && input.url;
    const method = init.method || (input && input.method) || 'GET';
    const body = init.body;
    const detail =
      body instanceof Blob ? `blob:${body.type || 'unknown'}:${body.size}` :
      body instanceof FormData ? 'form-data' :
      body instanceof ArrayBuffer ? `array-buffer:${body.byteLength}` :
      body && body.constructor ? body.constructor.name :
      '';
    add('fetch', method, url, detail);
    return originalFetch.apply(this, arguments);
  };

  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function patchedOpen(method, url) {
    this.__feishuSniffer = { method, url };
    return originalOpen.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function patchedSend(body) {
    const meta = this.__feishuSniffer || {};
    const detail =
      body instanceof Blob ? `blob:${body.type || 'unknown'}:${body.size}` :
      body instanceof FormData ? 'form-data' :
      body instanceof ArrayBuffer ? `array-buffer:${body.byteLength}` :
      body && body.constructor ? body.constructor.name :
      '';
    add('xhr', meta.method, meta.url, detail);
    return originalSend.apply(this, arguments);
  };

  window.__feishuUploadUrls = {
    records,
    hosts: () => [...new Set(records.map((r) => r.host))].sort(),
    table: () => console.table(records),
    copy: async () => {
      const text = JSON.stringify(records, null, 2);
      await navigator.clipboard.writeText(text);
      console.log(`[feishu-url-sniffer] copied ${records.length} records`);
    }
  };

  console.log('[feishu-url-sniffer] active. Reproduce paste/upload, then run: __feishuUploadUrls.table() or await __feishuUploadUrls.copy()');
})();
