/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

function Help(props) {
  const {config: siteConfig, language = ''} = props;
  const {baseUrl, docsUrl} = siteConfig;
  const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
  const langPart = `${language ? `${language}/` : ''}`;
  const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

  const supportLinks = [
    {
      content: `Apprenez-en plus en utilisant [la documentation sur ce site.](${docUrl(
        'doc1.html',
      )})`,
      title: 'Naviguer dans la doc',
    },
    {
      content: 'Posez des questions à propos de la documentation et du projet',
      title: 'Rejoindre la communauté',
    },
    {
      content: "Decouvrez les nouveautés du projet",
      title: 'Rester informé',
    },
  ];

  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer documentContainer postContainer">
        <div className="post">
          <header className="postHeader">
            <h1>Besoin d'aide ?</h1>
          </header>
          <p>Ce projet est maintenu par des personnes dévouées.</p>
          <GridBlock contents={supportLinks} layout="threeColumn" />
        </div>
      </Container>
    </div>
  );
}

module.exports = Help;
