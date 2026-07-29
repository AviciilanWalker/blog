// 记录每次访问
const { Redis } = require('@upstash/redis');
const UAParser = require('ua-parser-js');

const redis = Redis.fromEnv();

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const ip = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || 'unknown';

    // 屏蔽自己的 IP，不记录
    if (ip === '118.114.81.220') {
      return res.status(200).json({ ok: true, skipped: true });
    }

    // 解析路径：优先用客户端传来的，其次用 referer
    let path = '';
    try {
      const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
      path = body.path || '';
    } catch (_) {}
    if (!path) {
      try {
        const ref = req.headers.referer || '';
        path = ref ? new URL(ref).pathname : '';
      } catch (_) {}
    }

    // 不记录 visits 页面的访问
    if (path.startsWith('/visits')) {
      return res.status(200).json({ ok: true, skipped: true });
    }

    // 解析设备信息
    const uaParser = new UAParser(req.headers['user-agent'] || '');
    const ua = uaParser.getResult();

    const entry = {
      ip,
      country: req.headers['x-vercel-ip-country'] || '',
      region: req.headers['x-vercel-ip-country-region'] || '',
      city: req.headers['x-vercel-ip-city'] || '',
      lat: req.headers['x-vercel-ip-latitude'] || '',
      lon: req.headers['x-vercel-ip-longitude'] || '',
      path,
      browser: ua.browser.name ? `${ua.browser.name} ${ua.browser.version || ''}`.trim() : '',
      os: ua.os.name ? `${ua.os.name} ${ua.os.version || ''}`.trim() : '',
      device: ua.device.type ? `${ua.device.vendor || ''} ${ua.device.model || ''} ${ua.device.type}`.trim() : 'desktop',
      ua: (req.headers['user-agent'] || '').slice(0, 200),
      time: new Date().toISOString(),
    };

    const id = `v:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`;

    await redis.set(id, JSON.stringify(entry));
    await redis.expire(id, 60 * 60 * 24 * 30);

    await redis.incr('visits:total');

    res.status(200).json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
