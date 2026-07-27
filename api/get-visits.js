// 查询访问记录
const { Redis } = require('@upstash/redis');

const redis = Redis.fromEnv();

module.exports = async (req, res) => {
  if (req.query.secret !== 'requiem2026') {
    return res.status(403).json({ error: 'nope' });
  }

  try {
    const total = (await redis.get('visits:total')) || 0;

    let keys = [];
    let cursor = '0';
    do {
      const [next, found] = await redis.scan(cursor, { match: 'v:*', count: 100 });
      cursor = next;
      keys = keys.concat(found);
    } while (cursor !== '0' && keys.length < 200);

    keys.sort((a, b) => {
      const ta = parseInt(a.split(':')[1]) || 0;
      const tb = parseInt(b.split(':')[1]) || 0;
      return tb - ta;
    });

    const visits = [];
    for (const key of keys.slice(0, 100)) {
      const raw = await redis.get(key);
      if (raw) visits.push(typeof raw === 'string' ? JSON.parse(raw) : raw);
    }

    res.status(200).json({ total, visits });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
