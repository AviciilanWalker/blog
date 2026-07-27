// 记录每次访问到 Upstash Redis
const { Redis } = require('@upstash/redis');

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_URL,
  token: process.env.UPSTASH_REDIS_TOKEN,
});

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const ip = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || 'unknown';

    const entry = {
      ip,
      country: req.headers['x-vercel-ip-country'] || '',
      region: req.headers['x-vercel-ip-country-region'] || '',
      city: req.headers['x-vercel-ip-city'] || '',
      lat: req.headers['x-vercel-ip-latitude'] || '',
      lon: req.headers['x-vercel-ip-longitude'] || '',
      path: req.headers.referer || '',
      ua: (req.headers['user-agent'] || '').slice(0, 200),
      time: new Date().toISOString(),
    };

    const id = `v:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`;

    await redis.set(id, JSON.stringify(entry));
    await redis.expire(id, 60 * 60 * 24 * 30); // 30天过期

    // 更新计数器（方便查总量）
    await redis.incr('visits:total');

    res.status(200).json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
