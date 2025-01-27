/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["../templates/*.html", "../templates/**/*.html", "../static/css/tailwind.css"],
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
