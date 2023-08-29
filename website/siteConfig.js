/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

const path = require("path");

// List of projects/orgs using your project for the users page.
const users = [
  {
    caption: "jbesomi",
    // You will need to prepend the image path with your baseUrl
    // if it is not '/', like: '/test-site/img/image.jpg'.
    image: "/img/undraw_open_source.svg",
    infoLink: "https://texthero.org",
    pinned: true,
  },
];

const siteConfig = {
  title: "Texthero", // Title for your website.
  tagline:
    "Text preprocessing, representation and visualization from zero to hero.",
  url: "https://texthero.org", // Your website URL
  baseUrl: "/", // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: "texthero",
  organizationName: "jbesomi",
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    { doc: "getting-started", label: "Getting started" },
    { blog: true, label: "Tutorial" },
    { doc: "api-preprocessing", label: "API" },
    {
      href: "https://github.com/jbesomi/texthero",
      label: "GitHub",
    },
  ],

  blogSidebarTitle: { default: "Recent tutorials", all: "All tutorials posts" },

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */

  customDocsPath: path.basename(__dirname) + "/docs",

  usePrism: ["py"],
  highlight: {
    theme: "atom-one-dark",
  },

  /*
  headerIcon: 'img/logo_v2_transparent.png',
  footerIcon: 'img/favicon.ico',
  favicon: 'img/favicon.ico',
  */

  /* Colors for website */
  colors: {
    primaryColor: "#3f88c5",
    secondaryColor: "#ff8c42",
  },

  /* Custom fonts for website */
  /*
  fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },
  */

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Texthero - MIT license`,

  // Add custom scripts here that would be placed in <script> tags.
  scripts: [
    "https://buttons.github.io/buttons.js",
    "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
    "/js/code-block-buttons.js",
    "//cdnjs.cloudflare.com/ajax/libs/highlight.js/10.0.3/highlight.min.js",
    "/js/start_highlight.js",
    "https://www.googletagmanager.com/gtag/js?id=G-0V7XX3QG4C",
    "/js/analytics.js",
  ],
  stylesheets: ["/css/code-block-buttons.css", "/css/sphinx_basic.css"],

  // On page navigation for the current documentation page.
  onPageNav: "separate",
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: "img/T.png",
  twitterImage: "img/T.png",

  favicon: "img/favicon.png",

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',
};

module.exports = siteConfig;
