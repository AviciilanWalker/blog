---
layout: home

hero:
  name: "Requiem如是说"
  text: "请输入文本"
  tagline: 不知道为什么但就是很想急头白脸花100租个域名然后写一些莫名其妙的东西

features:
  - title: LLM 学习
    details: 我表弟也能看懂
    link: /guide/phase1
    linkText: 开始阅读
  - title: Vibe Coding
    details: 让 AI 帮你写代码，你只需要描述你想要什么
    link: /vibe-coding/
    linkText: 上手试试
  - title: 大学生存指南
    details: 怎么把大学四年过得爽、卷得赢
    link: /survival-guide/
    linkText: 进来看看
  - title: 一些随笔
    details: 踩过的坑，搞砸的事，错付的人
    link: /notes/
    linkText: 随便看看
---

<div class="secret-link" style="text-align:center;margin-top:2rem;">
  <a href="javascript:void(0)" style="color:#bbb;font-size:0.8rem;text-decoration:none;">想看点别的？</a>
</div>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const btn = document.querySelector('.secret-link a')
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
