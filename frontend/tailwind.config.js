/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Space Grotesk', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        ink: '#12263A',
        mist: '#F3F7FA',
        tide: '#4A7C8E',
        coral: '#EE6C4D',
      },
      boxShadow: {
        panel: '0 20px 40px -20px rgba(18, 38, 58, 0.35)',
      },
    },
  },
  plugins: [],
}