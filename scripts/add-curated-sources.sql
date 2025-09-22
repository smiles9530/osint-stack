-- Add curated, trustworthy sources from sources.json to the database
-- These sources are selected for their reliability, authority, and relevance

-- News Sources (Major International)
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('Reuters - All', 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best', 'News', 'en', true),
('BBC', 'https://feeds.bbci.co.uk/news/rss.xml', 'News', 'en', true),
('CNN', 'http://rss.cnn.com/rss/edition.rss', 'News', 'en', true),
('The Guardian', 'https://www.theguardian.com/rss', 'News', 'en', true),
('ABC News', 'https://abcnews.go.com/abcnews/topstories', 'News', 'en', true),
('CBC News', 'https://www.cbc.ca/cmlink/rss-topstories', 'News', 'en', true),
('Al Jazeera', 'https://www.aljazeera.com/xml/rss/all.xml', 'News', 'en', true),
('Channel News Asia', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml', 'News', 'en', true),
('South China Morning Post', 'https://www.scmp.com/rss/91/feed', 'News', 'en', true),
('All Africa', 'https://allafrica.com/tools/headlines/rdf/africa/headlines.rdf', 'News', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Technology Sources (High Authority)
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/index/', 'Tech', 'en', true),
('The Verge', 'https://www.theverge.com/rss/index.xml', 'Tech', 'en', true),
('Wired', 'https://www.wired.com/feed/rss', 'Tech', 'en', true),
('CNET', 'https://www.cnet.com/rss/news/', 'Tech', 'en', true),
('TechCrunch', 'https://techcrunch.com/feed/', 'Tech', 'en', true),
('Hacker News', 'https://hnrss.org/frontpage', 'Tech', 'en', true),
('Android Authority', 'https://www.androidauthority.com/feed/', 'Tech', 'en', true),
('BleepingComputer', 'https://www.bleepingcomputer.com/feed/', 'Tech', 'en', true),
('404 Media', 'https://www.404media.co/rss', 'Tech', 'en', true),
('9to5Linux', 'https://9to5linux.com/feed', 'Tech', 'en', true),
('Artificial Intelligence News', 'https://www.artificialintelligence-news.com/feed/', 'Tech', 'en', true),
('Biometric Update', 'https://www.biometricupdate.com/feed', 'Tech', 'en', true),
('Chip.pl', 'https://www.chip.pl/feed', 'Tech', 'pl', true),
('C++ Stories', 'https://www.cppstories.com/feed.xml', 'Tech', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Science Sources (Academic & Research)
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('Science Magazine', 'https://www.science.org/rss/news_current.xml', 'Science', 'en', true),
('ScienceDaily', 'https://www.sciencedaily.com/rss/all.xml', 'Science', 'en', true),
('Science News', 'https://www.sciencenews.org/feed', 'Science', 'en', true),
('Popular Science', 'https://www.popsci.com/feed', 'Science', 'en', true),
('Phys.org', 'https://phys.org/rss-feed', 'Science', 'en', true),
('Le Monde Science', 'https://www.lemonde.fr/en/science/rss_full.xml', 'Science', 'en', true),
('Nature Materials', 'https://www.nature.com/nmat/current_issue/rss', 'Science', 'en', true),
('Lateral with Tom Scott', 'https://audioboom.com/channels/5097784.rss', 'Science', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Security Sources (Cybersecurity)
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('DEF CON Announcements', 'https://defcon.org/defconrss.xml', 'Security', 'en', true),
('Hackaday', 'https://hackaday.com/feed', 'Security', 'en', true),
('HackRead', 'https://www.hackread.com/feed', 'Security', 'en', true),
('Security Affairs', 'https://securityaffairs.com/feed', 'Security', 'en', true),
('SecurityWeek', 'https://www.securityweek.com/feed', 'Security', 'en', true),
('Niebezpiecznik', 'https://feeds.feedburner.com/niebezpiecznik', 'Security', 'pl', true),
('Sekurak', 'https://sekurak.pl/rss', 'Security', 'pl', true),
('DPHacks', 'https://dphacks.com/feed', 'Security', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Business & Finance Sources
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('CNBC', 'https://www.cnbc.com/id/100003114/device/rss/rss.html', 'Business', 'en', true),
('Financial Times', 'https://www.ft.com/rss/home', 'Business', 'en', true),
('Bloomberg', 'https://feeds.bloomberg.com/markets/news.rss', 'Business', 'en', true),
('Wall Street Journal', 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml', 'Business', 'en', true),
('The Economist', 'https://www.economist.com/rss', 'Business', 'en', true),
('Business Insider', 'https://feeds.feedburner.com/businessinsider', 'Business', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Specialized Tech Communities
INSERT INTO sources (name, url, category, language, is_enabled) VALUES
('Reddit - Technology', 'https://www.reddit.com/r/technology/.rss', 'Tech', 'en', true),
('Reddit - Science', 'https://www.reddit.com/r/science/.rss', 'Science', 'en', true),
('Reddit - Programming', 'https://www.reddit.com/r/programming/.rss', 'Tech', 'en', true),
('Reddit - Cybersecurity', 'https://www.reddit.com/r/cybersecurity/.rss', 'Security', 'en', true),
('Reddit - Machine Learning', 'https://www.reddit.com/r/MachineLearning/.rss', 'Tech', 'en', true),
('Reddit - Linux', 'https://www.reddit.com/r/linux/.rss', 'Tech', 'en', true),
('Reddit - InternetIsBeautiful', 'https://www.reddit.com/r/InternetIsBeautiful/.rss', 'Tech', 'en', true)
ON CONFLICT (name) DO UPDATE SET 
    url = EXCLUDED.url,
    category = EXCLUDED.category,
    language = EXCLUDED.language,
    is_enabled = EXCLUDED.is_enabled;

-- Update statistics
ANALYZE sources;
