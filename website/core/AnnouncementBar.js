import React, {useState, useEffect} from 'react';

// import styles from './announcement-bar.css';

const STORAGE_DISMISS_KEY = 'docusaurus.announcement.dismiss';
const STORAGE_ID_KEY = 'docusaurus.announcement.id';

function AnnouncementBar() {


   //const {id, content, backgroundColor, textColor} = {
   //   id: 'supportus',
   //   content:
   //     '⭐️ If you like Docusaurus, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/facebook/docusaurus">GitHub</a>! ⭐️',
   // };

  const id = "supportus"
  const content = "⭐️ If you like Docusaurus"

  const [isClosed, setClosed] = useState(true);
  const handleClose = () => {
    localStorage.setItem(STORAGE_DISMISS_KEY, true);
    setClosed(true);
  };

  useEffect(() => {
    const viewedId = localStorage.getItem(STORAGE_ID_KEY);
    const isNewAnnouncement = id !== viewedId;

    localStorage.setItem(STORAGE_ID_KEY, id);

    if (isNewAnnouncement) {
      localStorage.setItem(STORAGE_DISMISS_KEY, false);
    }

    if (
      isNewAnnouncement ||
      localStorage.getItem(STORAGE_DISMISS_KEY) === 'false'
    ) {
      setClosed(false);
    }
  }, []);

  if (!content || isClosed) {
    return null;
  }

  return (
    <div
      className="announcementBar"
      style={{backgroundColor, color: textColor}}
      role="banner">
      <div
        className="announcementBarContent"
        dangerouslySetInnerHTML={{__html: content}}
      />

      <button
        type="button"
        className="announcementBarClose"
        onClick={handleClose}
        aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>
    </div>
  );
}

export default AnnouncementBar;
