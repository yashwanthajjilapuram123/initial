import streamlit as st

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import random

# Load the trained model and data files
with open("rf_pipeline_22.pkl", "rb") as f:
    rf_pipeline = pickle.load(f)

player_stats_recent = pd.read_pickle("player_recent_statistics.pkl")
head_to_head_stats = pd.read_pickle("head_to_head_stats.pkl")
venue_stats = pd.read_pickle("venue_stats.pkl")

# Predefined data for players and teams
teams = ["India", "South Africa", "Australia", "England", "West Indies","Sri Lanka","Pakistan","Bangladesh","Newzealand"]
team_players = {
    "India": ['AR Patel', 'Abhishek Sharma', 'Arshdeep Singh', 'Avesh Khan',
              'CV Varun', 'DJ Hooda', 'DL Chahar', 'Dhruv Jurel', 'HH Pandya',
              'HV Patel', 'Ishan Kishan', 'JJ Bumrah', 'JM Sharma', 'KK Ahmed',
              'Kuldeep Yadav', 'M Prasidh Krishna', 'MP Yadav', 'Mohammed Siraj',
              'Mukesh Kumar', 'Nithish Kumar Reddy', 'R Parag', 'R Sai Kishore',
              'RA Jadeja', 'RA Tripathi', 'RD Gaikwad', 'RG Sharma', 'RK Singh',
              'RR Pant', 'Ramandeep Singh', 'Ravi Bishnoi', 'S Dube', 'SA Yadav',
              'SS Iyer', 'SV Samson', 'Shahbaz Ahmed', 'Shivam Mavi',
              'Shubman Gill', 'TU Deshpande', 'Tilak Varma', 'Umran Malik',
              'V Kohli', 'Washington Sundar', 'YBK Jaiswal', 'YS Chahal'],
    "South Africa": ["A Nortje", "BC Fortuin", "D Brevis", "D Ferreira", "T Bavuma", "WD Parnell",
                     "AK Markram", "DA Miller", "H Klaasen", "K Rabada", "KA Maharaj"],
    "Australia": ['A Zampa', 'AC Agar', 'AJ Turner', 'AM Hardie', 'BJ Dwarshuis',
                  'BR McDermott', 'C Connolly', 'C Green', 'CJ Green', 'DA Warner',
                  'GJ Maxwell', 'J Fraser-McGurk', 'JP Behrendorff', 'JP Inglis',
                  'JR Hazlewood', 'JR Philippe', 'KW Richardson', 'MA Starc',
                  'MP Stoinis', 'MR Marsh', 'MS Wade', 'MW Short', 'NT Ellis',
                  'PJ Cummins', 'RP Meredith', 'SA Abbott', 'SH Johnson',
                  'SPD Smith', 'T Sangha', 'TH David', 'TM Head', 'XC Bartlett'],
    "England": ['AAP Atkinson', 'AU Rashid', 'BA Carse', 'BM Duckett', 'CJ Jordan',
                'CR Woakes', 'DJ Malan', 'DR Mousley', 'HC Brook', 'J Overton',
                'JA Turner', 'JC Archer', 'JC Buttler', 'JG Bethell',
                'JM Bairstow', 'JM Cox', 'L Wood', 'LS Livingstone', 'MA Wood',
                'MM Ali', 'PD Salt', 'RJW Topley', 'Rehan Ahmed', 'S Mahmood',
                'SM Curran', 'TS Mills', 'WG Jacks'],
    "West Indies": ['A Athanaze', 'AD Russell', 'ADS Fletcher', 'AJ Hosein',
                    'AS Joseph', 'BA King', 'E Lewis', 'FA Allen', 'G Motie',
                    'HR Walsh', 'J Charles', 'JO Holder', 'KR Mayers', 'MW Forde',
                    'N Pooran', 'O Thomas', 'OC McCoy', 'OF Smith', 'R Powell',
                    'R Shepherd', 'RA Reifer', 'RL Chase', 'S Joseph', 'SD Hope',
                    'SE Rutherford', 'SK Springer', 'SO Hetmyer', 'SS Cottrell',
                    'T Hinds'],
    "Sri Lanka" : ['A Dananjaya', 'AD Mathews', 'AM Fernando', 'B Fernando',
       'BKG Mendis', 'C Karunaratne', 'C Wickramasinghe', 'CAK Rajitha',
       'CBRLS Kumara', 'D Madushanka', 'DM de Silva', 'DN Wellalage',
       'KIC Asalanka', 'M Pathirana', 'M Theekshana', 'MD Shanaka',
       'MDKJ Perera', 'MNK Fernando', 'N Thushara', 'P Nissanka',
       'PBB Rajapaksa', 'PHKD Mendis', 'PM Liyanagamage', 'PVD Chameera',
       'PWH de Silva', 'RTM Mendis', 'S Samarawickrama', 'WIA Fernando'],
    "Pakistan" : ['Aamer Jamal', 'Abbas Afridi', 'Abrar Ahmed', 'Agha Salman',
       'Arafat Minhas', 'Arshad Iqbal', 'Asif Ali', 'Azam Khan',
       'Babar Azam', 'Faheem Ashraf', 'Fakhar Zaman', 'Haider Ali',
       'Haris Rauf', 'Hasan Ali', 'Haseebullah Khan', 'Iftikhar Ahmed',
       'Ihsanullah', 'Imad Wasim', 'Irfan Khan', 'Jahandad Khan',
       'Khushdil Shah', 'Mirza Baig', 'Mohammad Amir', 'Mohammad Haris',
       'Mohammad Nawaz', 'Mohammad Rizwan', 'Mohammad Wasim',
       'Naseem Shah', 'Omair Yousuf', 'Qasim Akram', 'Rohail Nazir',
       'Sahibzada Farhan', 'Saim Ayub', 'Shadab Khan',
       'Shaheen Shah Afridi', 'Shan Masood', 'Sufiyan Muqeem',
       'Tayyab Tahir', 'Usama Mir', 'Usman Khan', 'Zaman Khan'],
    "Bangladesh" : ['Afif Hossain', 'Hasan Mahmud', 'Hasan Murad', 'Jaker Ali',
       'Liton Das', 'Mahedi Hasan', 'Mahmudul Hasan Joy', 'Mahmudullah',
       'Mehedi Hasan Miraz', 'Mohammad Saifuddin', 'Mrittunjoy Chowdhury',
       'Mustafizur Rahman', 'Nasum Ahmed', 'Nazmul Hossain Shanto',
       'Parvez Hossain Emon', 'R Mondol', 'Rakibul Hasan',
       'Rishad Hossain', 'Rony Talukdar', 'Saif Hassan',
       'Shahadat Hossain', 'Shakib Al Hasan', 'Shamim Hossain',
       'Shoriful Islam', 'Soumya Sarkar', 'Sumon Khan', 'Tanvir Islam',
       'Tanzid Hasan', 'Tanzim Hasan Sakib', 'Taskin Ahmed',
       'Towhid Hridoy', 'Yasir Ali Chowdhury', 'Zakir Hasan'],
    "New Zealand" : ['A Ashok', 'AF Milne', 'BG Lister', 'BM Tickner', 'BV Sears',
       'CE McConchie', 'CJ Bowes', 'D Cleaver', 'D Foxcroft',
       'DJ Mitchell', 'DP Conway', 'FH Allen', 'GD Phillips',
       'HB Shipley', 'HM Nicholls', 'IS Sodhi', 'JA Clarkson', 'JA Duffy',
       'JDS Neesham', 'KA Jamieson', 'KS Williamson', 'LH Ferguson',
       'MG Bracewell', 'MJ Hay', 'MJ Henry', 'MJ Santner', 'MS Chapman',
       'R Ravindra', 'TA Blundell', 'TA Boult', 'TB Robinson',
       'TG Southee', 'TL Seifert', 'TWM Latham', "W O'Rourke", 'WA Young',
       'ZGF Foulkes']
}

# Title
st.title("🏏Yash's  Dream11 Team Selector")

# Sidebar Inputs
st.sidebar.header("Match Setup")
pitch_type = st.sidebar.selectbox("Select Pitch Type", ["Green", "Flat", "Dusty", "Bouncy", "Other"])
weather_conditions = st.sidebar.selectbox("Select Weather Conditions", ["Sunny", "Windy", "Overcast", "Humid"])
venue = st.sidebar.selectbox("Enter Venue", venue_stats['venue'].unique())

# Team Selection
team_a = st.sidebar.selectbox("Select Team A", teams)
team_b = st.sidebar.selectbox("Select Team B", [team for team in teams if team != team_a])

# Player Selection
team_a_players = st.sidebar.multiselect(f"Select Playing XI for {team_a}", team_players[team_a])
team_b_players = st.sidebar.multiselect(f"Select Playing XI for {team_b}", team_players[team_b])

# Validate Player Selection
if len(team_a_players) != 11 or len(team_b_players) != 11:
    st.error("Please select exactly 11 players for both Team A and Team B.")
    st.stop()

# Combine Teams into DataFrame
selected_players = pd.DataFrame({'player': team_a_players + team_b_players, 'team': [team_a] * 11 + [team_b] * 11})

# Ensure required columns exist
required_columns = ['total_runs', 'balls_faced', 'boundaries', 'sixes', 'strike_rate',
                    'batting_average', 'total_wickets', 'balls_bowled', 'lbw', 'bowled',
                    'runs_conceded', 'economy', 'bowling_average', 'catches_total',
                    'runouts_total', 'stumps_total', 'caught_and_bowled', 'catch_wickets', 'role', 'team']

for col in required_columns:
    if col not in player_stats_recent.columns:
        player_stats_recent[col] = 0 if col in required_columns[:-2] else "Unknown"

# Filter selected player stats
player_stats = player_stats_recent[player_stats_recent['player'].isin(selected_players['player'])]

# Predict Fantasy Points
player_stats['predicted_points'] = rf_pipeline.predict(player_stats[required_columns])

# Generate 3 Different Teams
num_teams = 3
teams = []

for _ in range(num_teams):
    team = player_stats.sample(n=11, weights=player_stats['predicted_points'], random_state=random.randint(1, 1000))
    teams.append(team)

# Assign Trump Card Player
for team in teams:
    trump_card = team.iloc[len(team) // 2]  # Middle player in ranking
    team['is_trump_card'] = team['player'] == trump_card['player']

# Probabilistic Captain & Vice-Captain Selection
for team in teams:
    top_5_candidates = team.head(min(5, len(team)))  # Ensure at least 5 players available
    if len(top_5_candidates) == 0:
        continue

    probabilities = [0.4, 0.3, 0.2, 0.1, 0.05][:len(top_5_candidates)]  # Adjust probability size

    captain = random.choices(top_5_candidates['player'].tolist(), weights=probabilities, k=1)[0]
    team['is_captain'] = team['player'] == captain

    vice_captain_candidates = top_5_candidates[top_5_candidates['player'] != captain]
    if len(vice_captain_candidates) > 0:
        vice_captain = random.choices(vice_captain_candidates['player'].tolist(), weights=probabilities[1:], k=1)[0]
        team['is_vice_captain'] = team['player'] == vice_captain
    else:
        team['is_vice_captain'] = False

# Display Teams
#st.subheader("Optimized Dream11 Teams")
st.subheader("మీ పైసల్ పోతే నాకు తెల్వదు కానీ మీకు కోటి రూపాయాలు వస్తే నాకు సగం")
for i, team in enumerate(teams, start=1):
    st.write(f"### Team {i}")
    st.dataframe(team[['player', 'team', 'role', 'predicted_points', 'is_trump_card', 'is_captain', 'is_vice_captain']])
# Pitch Type Recommendations
pitch_recommendations = {
    "Green": "✅ This pitch favors **Bowlers**, especially fast bowlers.",
    "Flat": "✅ This pitch favors **Batters**, making it a high-scoring game.",
    "Dusty": "✅ This pitch supports **Spinners** and **All-Rounders**.",
    "Bouncy": "✅ This pitch helps **Fast Bowlers** with extra pace and bounce.",
    "Other": "✅ This pitch has a **neutral effect**, favoring all players equally."
}

# Weather Recommendations
weather_recommendations = {
    "Sunny": "☀️ **Good conditions for Batters**. The ball comes onto the bat nicely.",
    "Windy": "🌬️ **Favors Fast Bowlers**. Swing bowlers will be more effective.",
    "Overcast": "☁️ **Favors Swing & Seam Bowlers**. The ball will move a lot in the air.",
    "Humid": "🌡️ **Helps Spinners & Swing Bowlers** due to moisture in the air."
}
# Display Pitch Recommendation
st.subheader("🏟️ Pitch Type Impact")
st.write(pitch_recommendations[pitch_type])

# Display Weather Recommendation
st.subheader("🌦️ Weather Impact")
st.write(weather_recommendations[weather_conditions])
