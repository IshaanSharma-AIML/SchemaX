// tailwind.config.js

/** @type {import('tailwindcss').Config} */
const config = {
  darkMode: 'class',
    content: [
      "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
      "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
      "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
      extend: {
        colors: {
          ai: {
            teal: '#2dd4bf',
            blue: '#38bdf8',
            purple: '#8b5cf6',
          }
        },
        animation: {
          'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        },
      },
    },
    plugins: [],
  };
  
  module.exports = config;