/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["../templates/*.html", "../templates/**/*.html", "../static/css/tailwind.css"],
    // these colors are defined in a python utils function and are not scanned for tailwind classes
    safelist: ["bg-emerald-400", "bg-blue-400", "bg-purple-400", "bg-orange-400", "bg-slate-400"],
    theme: {
        extend: {
            fontFamily: {
                sans: ["IBM Plex Mono", "sans-serif"], // Set 'Roboto' as the default sans font
                serif: ["IBM Plex Mono", "serif"],
                mono: ["IBM Plex Mono", "monospace"],
            },
        },
    },
    plugins: [],
};
