---
title: "想告诉你的一些事"
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  if (!window.location.search.includes('p=1') && !sessionStorage.getItem('secret_access')) {
    window.location.replace('/')
  } else {
    sessionStorage.setItem('secret_access', '1')
  }
})
</script>

# 想告诉你的一些事

一个隐藏的角落。一些想说但还没当面说的话。

---

### 文章列表

- **[认知、过拟合与一些不成器的恨](./2026-07-22)** —— 2026 年 7 月 22 日

---

*这里会慢慢变多的。*
