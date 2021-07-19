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
              width={160}
              height={30}
              title="GitHub Stars"
              style={{margin: "20px"}}
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


const HomeBox = props => (
    <div class={"homebox " + props.position}>
        {props.children}
    </div>
)


const Separator = () => (
    <div className="home_separator"></div>
)


const Code = props => (
    <pre>
        <code className="python">
            {props.children}
        </code>
    </pre>
)

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

            <Separator />

            <div class="showcase">

                <HomeBox position="left">
                    <h2 className="projectTitle">
                        <small>Import texthero ... </small>
                    </h2>
                    <Code>{`import texthero as hero
import pandas as pd`
                    }</Code>
                </HomeBox>

                <HomeBox position="right">
                    <h2 className="projectTitle">
                        <small>... load any text dataset with Pandas</small>
                    </h2>
                    <Code>{`df = pd.read_csv(
    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)
df.head(2)`}</Code>
                    
                    <table border="1" class="dataframe">
                      <thead>
                        <tr>
                          <th></th>
                          <th>text</th>
                          <th>topic</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>0</th>
                          <td>Claxton hunting first major medal\n\nBritish h...</td>
                          <td>athletics</td>
                        </tr>
                        <tr>
                          <th>1</th>
                          <td>O'Sullivan could run in Worlds\n\nSonia O'Sull...</td>
                          <td>athletics</td>
                        </tr>
                      </tbody>
                    </table>

                </HomeBox>

                <HomeBox position="left">
                    <h2 className="projectTitle">
                        <small>Preprocess it ...</small>
                    </h2>
                    <Code>{`df['text'] = hero.clean(df['text'])`}
                    
                    </Code>

                    <table border="1" class="dataframe">
                      <thead>
                        <tr>
                          <th></th>
                          <th>text</th>
                          <th>topic</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>0</th>
                          <td>claxton hunting first major medal british hurd...</td>
                          <td>athletics</td>
                        </tr>
                        <tr>
                          <th>1</th>
                          <td>sullivan could run worlds sonia sullivan indic...</td>
                          <td>athletics</td>
                        </tr>
                      </tbody>
                    </table>

                    <MarkdownBlock>
                    > Look at the [preprocessing API](/docs/api-preprocessing) for more customization 
                    </MarkdownBlock>


                </HomeBox>


                <HomeBox position="right">
                    <h2 className="projectTitle">
                        <small>... represent it</small>
                    </h2>
                    <Code>{`df['tfidf'] = (
    hero.tfidf(df['text'], max_features=100)
)
df[["tfidf", "topic"]].head(2)
                        `}
                    
                    </Code>

                    <table border="1" class="dataframe">
                      <thead>
                        <tr>
                          <th></th>
                          <th>tfidf</th>
                          <th>topic</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>0</th>
                          <td>[0.0, 0.13194458247285848, 0.0, 0.0, 0.0, 0.0,...</td>
                          <td>athletics</td>
                        </tr>
                        <tr>
                          <th>1</th>
                          <td>[0.0, 0.13056235989725676, 0.0, 0.205187581391...</td>
                          <td>athletics</td>
                        </tr>
                      </tbody>
                    </table>

                    <MarkdownBlock>
                    > There are [many other ways](/docs/api-representation) to represent the data
                    </MarkdownBlock>

                </HomeBox>




                <HomeBox position="left">
                    <h2 className="projectTitle">
                        <small>Reduce dimension and visualize the vector space</small>
                    </h2>
                    <Code>{`df['pca'] = hero.pca(df['tfidf'])
hero.scatterplot(
    df, 
    col='pca', 
    color='topic', 
    title="PCA BBC Sport news"
)`}
                    
                    </Code>
                    
                    <img src="/img/scatterplot_bccsport.svg" alt="" />

                </HomeBox>


                <HomeBox position="right">
                    <h2 className="projectTitle">
                        <small>... need more? find named entities</small>
                    </h2>
                    <Code>{`df['named_entities'] = (
    hero.named_entities(df['text'])
)
df[['named_entities', 'topic']].head(2)
                        `}
                    
                    </Code>

                    <table border="1" class="dataframe">
                      <thead>
                        <tr>
                          <th></th>
                          <th>named_entities</th>
                          <th>topic</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>0</th>
                          <td>[(claxton, ORG, 0, 7), (first, ORDINAL, 16, 21...</td>
                          <td>athletics</td>
                        </tr>
                        <tr>
                          <th>1</th>
                          <td>[(sullivan, ORG, 0, 8), (sonia sullivan, PERSO...</td>
                          <td>athletics</td>
                        </tr>
                      </tbody>
                    </table>

                </HomeBox>




                <HomeBox position="left">
                    <h2 className="projectTitle">
                        <small>Show top words ...</small>
                    </h2>
                    <Code>{`NUM_TOP_WORDS = 5
hero.top_words(df['text'])[:NUM_TOP_WORDS]
                        `}
                    
                    </Code>

                        <table border="1" class="dataframe">
                          <thead>
                            <tr>
                              <th></th>
                              <th>text</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <th>said</th>
                              <td>1338</td>
                            </tr>
                            <tr>
                              <th>first</th>
                              <td>790</td>
                            </tr>
                            <tr>
                              <th>england</th>
                              <td>749</td>
                            </tr>
                            <tr>
                              <th>game</th>
                              <td>681</td>
                            </tr>
                            <tr>
                              <th>one</th>
                              <td>671</td>
                            </tr>
                          </tbody>
                        </table>

                </HomeBox>


                <HomeBox position="right">
                    <h2 className="projectTitle">
                        <small>And much more !</small>
                    </h2>
                </HomeBox>


            </div>

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
