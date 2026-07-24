---
title: "嘘——"
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  if (window.location.search.includes('p=1') || sessionStorage.getItem('secret_access')) {
    sessionStorage.setItem('secret_access', '1')
    window.location.replace('/sth-to-tell-u/')
  } else {
    window.location.replace('/')
  }
})
</script>
