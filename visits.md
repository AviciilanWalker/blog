---
title: "访客记录"
---

<script setup>
import { ref, onMounted } from 'vue'

const authed = ref(false)
const pwdInput = ref('')
const error = ref('')
const visits = ref([])
const total = ref(0)
const loading = ref(true)

function checkPassword() {
  if (pwdInput.value === 'zhangmin060320') {
    authed.value = true
    loadVisits()
  } else {
    error.value = '密码不对'
  }
}

async function loadVisits() {
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
}

function formatTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function location(v) {
  const parts = [v.country, v.region, v.city].filter(Boolean)
  return parts.join(' ') || '未知'
}
</script>

<div v-if="!authed" style="max-width:300px;margin:40px auto;">
  <h2>访客记录</h2>
  <input
    v-model="pwdInput"
    type="password"
    placeholder="输入密码"
    style="width:100%;padding:10px;font-size:1rem;border:1px solid var(--vp-c-divider);border-radius:6px;background:var(--vp-c-bg);color:var(--vp-c-text-1);"
    @keyup.enter="checkPassword"
  />
  <button
    @click="checkPassword"
    style="width:100%;margin-top:10px;padding:10px;font-size:1rem;border:none;border-radius:6px;background:var(--vp-c-brand);color:#fff;cursor:pointer;"
  >进入</button>
  <p v-if="error" style="color:#e74c3c;margin-top:10px;text-align:center;">{{ error }}</p>
</div>

<div v-else>
  <h1>谁来过</h1>
  <div v-if="loading">加载中...</div>
  <div v-else-if="error" style="color:#e74c3c">出错了：{{ error }}</div>
  <div v-else>
    <p>共 <strong>{{ total }}</strong> 条记录（最近 100 条）</p>
    <table>
      <thead>
        <tr>
          <th>时间</th>
          <th>IP</th>
          <th>位置</th>
          <th>页面</th>
          <th>设备</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="v in visits" :key="v.ip + v.time">
          <td style="white-space:nowrap;font-size:0.85em;">{{ formatTime(v.time) }}</td>
          <td style="font-family:monospace;font-size:0.82em;">{{ v.ip }}</td>
          <td style="font-size:0.85em;">{{ location(v) }}</td>
          <td style="font-family:monospace;font-size:0.82em;word-break:break-all;">{{ v.path || '/' }}</td>
          <td style="font-size:0.78em;max-width:180px;">
            {{ v.device !== 'desktop' && v.device ? v.device : '' }}
            {{ v.browser || '' }}
            {{ v.os || '' }}
          </td>
        </tr>
      </tbody>
    </table>
    <p v-if="visits.length === 0" style="color:#999">还没有人来过。</p>
  </div>
</div>

<style scoped>
table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--vp-c-divider); }
th { font-weight: 600; color: var(--vp-c-text-2); font-size: 0.75em; text-transform: uppercase; white-space: nowrap; }
tr:hover { background: var(--vp-c-bg-soft); }
</style>
