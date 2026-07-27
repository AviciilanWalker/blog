import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: "Requiem如是说",
  description: "从LLM到人生，想到什么写什么",
  lang: 'zh-CN',

  head: [
    ['meta', { name: 'robots', content: 'noindex, nofollow' }]
  ],

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '学习笔记', link: '/guide/phase1' },
      { text: 'Vibe Coding', link: '/vibe-coding/' },
      { text: '随笔', link: '/notes/' },
      { text: '大学生存指南', link: '/survival-guide/' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'LLM 学习体系',
          items: [
            { text: '第一阶段：建立直觉', link: '/guide/phase1' },
            { text: '第二阶段：理解原理', link: '/guide/phase2' },
            { text: '第三阶段：训练过程', link: '/guide/phase3' },
            { text: '第四阶段：使用与工程', link: '/guide/phase4' },
            { text: '第五阶段：进阶主题', link: '/guide/phase5' },
            { text: '底层数学概念速查', link: '/guide/math-ref' },
            { text: '计算机底层概念速查', link: '/guide/cs-ref' },
          ]
        }
      ],
      '/vibe-coding/': [
        {
          text: 'Vibe Coding',
          items: [
            { text: '入门：让 AI 帮你写代码', link: '/vibe-coding/' },
          ]
        }
      ],
      '/survival-guide/': [
        {
          text: '大学生存指南',
          items: [
            {
              text: '活得更爽',
              collapsed: false,
              items: [
                { text: '概述', link: '/survival-guide/live-better/' },
              ]
            },
            {
              text: '卷得更猛',
              collapsed: false,
              items: [
                { text: '概述', link: '/survival-guide/hustle-harder/' },
              ]
            },
          ]
        }
      ],
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present Requiem'
    },

    search: {
      provider: 'local'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ]
  }
})
