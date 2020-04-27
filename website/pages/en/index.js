const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');
const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

//const AnnouncementBar = require(process.cwd() + '/core/AnnouncementBar.js');

// import AnnouncementBar from '/core/AnnouncementBar.js'



const siteConfig = require(process.cwd() + '/siteConfig.js');

class Button extends React.Component {
  render() {
    return (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={this.props.href} target={this.props.target}>
          {this.props.children}
        </a>
      </div>
    );
  }
}

function assetUrl(img) {
  return siteConfig.baseUrl + 'docs/assets/' + img;
}

function docUrl(doc) {
  return siteConfig.baseUrl + 'docs/' + doc;
}

Button.defaultProps = {
  target: '_self',
};

const SplashContainer = props => (
  <div className="homeContainer">
    <div className="homeSplashFade">
      <div className="wrapper homeWrapper">{props.children}</div>
    </div>
  </div>
);

const Logo = props => (
  <div className="projectLogo">
    <img src={props.img_src} />
  </div>
);

const ProjectTitle = props => (

  <div>
    <h2 className="projectTitle">
      <small>{siteConfig.tagline}</small>
    </h2>

    <h3>
      Texthero is a python package to work with text data <strong>efficiently</strong>. <br />
      It empowers NLP developers with a tool to quickly understand any text-based dataset and <br />
      it provides a solid pipeline to clean and represent text data, from zero to hero.
    </h3>
  </div>

);

const PromoSection = props => (
  <div className="section promoSection">
    <div className="promoRow">
      <div className="pluginRowBlock">{props.children}</div>
    </div>
  </div>
);

const GithuButton = props => (
   <iframe
              src="https://ghbtns.com/github-btn.html?user=jbesomi&amp;repo=texthero&amp;type=star&amp;count=true&amp;size=large"
              frameBorder={0}
              scrolling={0}
              width={120}
              height={30}
              title="GitHub Stars"
            />
);

const Introduction = props => (
      <div className="wrapper homeWrapper">
         <div className="inner">
            <MarkdownBlock>
            </MarkdownBlock>
         </div>
      </div>
)

const Block = props => (
  <Container
    padding={['bottom', 'top']}
    id={props.id}
    background={props.background}>
    <GridBlock
      align="center"
      contents={props.children}
      layout={props.layout}
    />
  </Container>
);

const Features = () => (
   <Block layout="fourColumn">
     {[
       {
         content: 'This is the content of my feature',
         image: `${siteConfig}img/undraw_react.svg`,
         imageAlign: 'top',
         title: 'Feature One',
       },
       {
         content: 'The content of my second feature',
         image: `${siteConfig}img/undraw_operating_system.svg`,
         imageAlign: 'top',
         title: 'Feature Two',
       },
     ]}
   </Block>
)

  /* <AnnouncementBar /> */

class HomeSplash extends React.Component {



  render() {
    return (
      <SplashContainer>

        <Logo img_src={assetUrl('texthero.png')} />
        <div className="inner">
          <ProjectTitle />
          <PromoSection>
            <Button href={docUrl('getting-started')} >Getting started</Button>
            <Button href={siteConfig.baseUrl + 'blog'}>Tutorial</Button>
            <Button href={docUrl('api-preprocessing')}>API</Button>
            <Button href='https://github.com/jbesomi/texthero'>Github</Button>
          </PromoSection>
          <GithuButton />
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    return (
      <div>
        <HomeSplash />
        <div className="mainContainer">
           <Introduction />
        </div>
      </div>
    );
  }
}

module.exports = Index;
