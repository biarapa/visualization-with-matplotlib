import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime
import matplotlib.colors as colors

tweets = pd.read_csv('~/Downloads/analyze-reviews/tweets.csv')

#make a candidate column based on tweets on text column
def get_candidate(row):
    candidates = []
    text = row['text'].lower()          #make a low case
    if 'clinton' in text or 'hillary' in text:
        candidates.append('clinton')
    if 'trump' in text or 'donald' in text:
        candidates.append('trump')
    if 'sanders' in text or 'bernie' in text:
        candidates.append('sanders')
    return ','.join(candidates)
tweets['candidate'] = tweets.apply(get_candidate, axis=1)

#make a bar chart
#counts = tweets['candidate'].value_counts()
#plt.bar(range(len(counts)), counts)

#customize chart
tweets['created'] = pd.to_datetime(tweets['created'])
tweets['user_created'] = pd.to_datetime(tweets['user_created'])

tweets['user_age'] = tweets['user_created'].apply(lambda x: (datetime.now() - x).total_seconds()/3600/24/365)
#plt.hist(tweets['user_age'])

#add labels
#plt.title('Tweets mentioning candidates')
#plt.xlabel('Twitter account age in years')
#plt.ylabel('# of tweets')

#make a stacked histogram
clinton_tweets = tweets['user_age'][tweets['candidate'] == 'clinton']
sanders_tweets = tweets['user_age'][tweets['candidate'] == 'sanders']
trump_tweets = tweets['user_age'][tweets['candidate'] == 'trump']
#plt.hist([clinton_tweets, sanders_tweets, trump_tweets],
#            stacked = True,
#            label = ['clinton', 'sanders', 'trump'])
#plt.legend()
#plt.title("Tweets mentioning each candidate")
#plt.xlabel("Twitter account age in years")
#plt.ylabel('# of tweets')

#annotate the histogram
plt.hist([clinton_tweets, sanders_tweets, trump_tweets],
            stacked=True,
            label=['clinton', 'sanders', 'trump'])
plt.legend()
plt.title('Tweets mentioning each candidate')
plt.xlabel("Twitter account age in years")
plt.ylabel('# of tweets')
plt.annotate('More trump tweets', xy=(1, 10000), xytext=(2, 10000),         #xy = x & y coordinates where the arrow should start. xytest = x & y coordinates where the text should start
                arrowprops=dict(facecolor='black'))                         #arrowprops = customize about arrow, example color

#multiple subplots
tweets['red'] = tweets['user_bg_color'].apply(lambda x: colors.hex2color('#{0}'.format(x))[0])
tweets['blue'] = tweets['user_bg_color'].apply(lambda x: colors.hex2color('#{0}'.format(x))[2])


#remove common background colors
tweets['user_bg_color'].value_counts()
tc = tweets[~tweets['user_bg_color'].isin(['C0DEED', 'OOOOOO', 'F5F8FA'])]
def create_plot(data):
    fig, axes = plt.subplots(nrows=2, ncols=2)      #will generate 2x2 grid of axes objects
    ax0, ax1, ax2, ax3 = axes.flat

    ax0.hist(tweets['red'])
    ax0.set_title('Red in backgrounds')

    ax1.hist(tweets['red'][tweets['candidate']=='trump'].values)
    ax1.set_title('Red in Trump tweeters')

    ax2.hist(tweets['blue'])
    ax2.set_title('Blue in backgrounds')

    ax3.hist(tweets['blue'][tweets['candidate']=='trump'].values)
    ax3.set_title('Blue in Trump tweeters')
    #plt.tight_layout()              #method to reduce padding in the graphs & fit all the elements
create_plot(tc)

#plotting sentiment
gr = tweets.groupby('candidate').agg([np.mean, np.std])

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))          #7 inches by 7 inches
ax0, ax1 = axes.flat

std = gr['polarity']['std'].iloc[1:]
mean = gr['polarity']['mean'].iloc[1:]

ax0.bar(range(len(std)), std)
ax0.set_xticklabels(std.index, rotation=45)                 #set tick labels & rotate labels 45 degress
ax0.set_title('Standard deviation of tweet sentiment')

ax1.bar(range(len(mean)), mean)
ax1.set_xticklabels(mean.index, rotation=45)
ax1.set_title('Mean tweet sentiment')

plt.tight_layout()              #method to reduce padding in the graphs & fit all the elements
plt.show()

#generating a side by side bar plot
def tweet_lengths(text):
    if len(text) < 100:
        return 'short'
    elif 100 <= len(text) <= 135:
        return 'medium'
    else:
        return 'long'
tweets['tweet_length'] = tweets['text'].apply(tweet_lengths)

tl = {}
for candidate in ['clinton', 'sanders', 'trump']:
    tl[candidate] = tweets['tweet_length'][tweets['candidate']==candidate].value_counts()

fig, ax = plt.subplots()
width = .5
x = np.array(range(0, 6, 2))                            #generate a sequence values. each value is the start of a category
ax.bar(x, tl['clinton'], width, color='g')              #plot clinton tweets on the Axes object, with the bars at the positions defined by x
ax.bar(x + width, tl['sanders'], width, color='b')      #plot sanders's on the Axes object, add width to x to move the bars to the right
ax.bar(x + (width*2), tl['trump'], width, color='r')    #plot trump's, add width*2 to x to move the bars to the far right

ax.set_ylabel('# of tweets')
ax.set_title('Number of Tweets per candidate by length')
ax.set_xticks(x + (width*1.5))
ax.set_xticklabels(('long', 'medium', 'short'))
ax.set_xlabel('Tweet length')
plt.show()