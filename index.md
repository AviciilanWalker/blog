---
layout: home

hero:
  name: "Requiem如是说"
  text: "请输入文本"
  tagline: 不知道为什么但就是很想急头白脸花100租个域名然后写一些莫名其妙的东西
  actions:
    - theme: brand
      text: LLM 学习笔记
      link: /guide/phase1
    - theme: alt
      text: 技术随笔
      link: /notes/
    - theme: alt
      text: 想看点别的？
      link: 'javascript:void(0)'

features:
  - title: LLM 学习
    details: 我表弟也能看懂
    link: /guide/phase1
    linkText: 开始阅读
  - title: 一些随笔
    details: 踩过的坑，搞砸的事，错付的人
    link: /notes/
    linkText: 随便看看
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const btn = document.querySelector('a[href="javascript:void(0)"]')
  if (btn) {
    btn.addEventListener('click', (e) => {
      e.preventDefault()
      const first = prompt('你是谁？')
      if (first !== '刘芯琳') {
        if (first !== null) alert('不认识你，别闹。')
        return
      }
      const second = prompt('我又是谁？')
      if (second === '王敏璋') {
        sessionStorage.setItem('secret_access', '1')
        window.location.href = '/sth-to-tell-u/?p=1'
      } else if (second !== null) {
        alert('这样也好')
      }
    })
  }
})
</script>
