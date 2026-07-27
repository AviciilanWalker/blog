---
title: "访客记录"
---

<script setup>
import { ref, onMounted } from 'vue'

const visits = ref([])
const total = ref(0)
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    const res = await fetch('/api/get-visits?secret=requiem2026')
    if (!res.ok) throw new Error('加载失败')
    const data = await res.json()
    visits.value = data.visits || []
    total.value = data.total || 0
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function location(v) {
  const parts = [v.country, v.region, v.city].filter(Boolean)
  return parts.join(' / ') || '未知'
}

function uaShort(ua) {
  if (!ua) return ''
  // 提取浏览器和系统
  const m = ua.match(/\((.*?)\)/)
  return m ? m[1].split(';').slice(-1)[0]?.trim() || ua.slice(0, 60) : ua.slice(0, 40)
}
</script>

# 谁来过

<div v-if="loading">加载中...</div>
<div v-else-if="error">出错了：{{ error }}</div>
<div v-else>
  <p>共 <strong>{{ total }}</strong> 条记录（最近 100 条）</p>
  <table>
    <thead>
      <tr>
        <th>时间</th>
        <th>IP</th>
        <th>位置</th>
        <th>设备</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="v in visits" :key="v.ip + v.time">
        <td style="white-space:nowrap">{{ formatTime(v.time) }}</td>
        <td style="font-family:monospace;font-size:0.85em">{{ v.ip }}</td>
        <td>{{ location(v) }}</td>
        <td style="font-size:0.8em;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ uaShort(v.ua) }}</td>
      </tr>
    </tbody>
  </table>
  <p v-if="visits.length === 0" style="color:#999">还没有人来看过。</p>
</div>

<style scoped>
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--vp-c-divider); }
th { font-weight: 600; color: var(--vp-c-text-2); font-size: 0.8em; text-transform: uppercase; }
tr:hover { background: var(--vp-c-bg-soft); }
</style>
