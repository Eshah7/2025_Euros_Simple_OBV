#Import Required Libraries 
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from statsbombpy import sb

import warnings
from statsbombpy.api_client import NoAuthWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NoAuthWarning)

#Constants 
#UEFA Women's Euros 2025 
COMPETITION_ID = 53
SEASON_ID = 315 

#UI Constants 
st.set_page_config(page_title = "⚽ Women's Euros 2025", layout = "wide")
st.title("Women's 2025 Euros - :blue[On-Ball Value Model]")
st.write("Imagine yourself on the soccer field... The ball is finally passed to you and your next decision will impact the game positively or negatively. You pass the ball across the half line, and your team scores a goal after a few more passes. So, how much did you contribute to that goal? Did you increase the probability of your team being able to score a goal?")
st.write("The On-Ball Value (OBV) model developed by StatsBomb looks at the difference in the model's predicted scoring probability before and after interaction with the ball, such as a pass or a shot. If you pass a ball near the net, then you are increasing the team's chance of scoring.")
st.write("We will utilize the 2025 Euros data to train our own model and determine the players that contributed to their team's goals the most. For this particular analysis, we will be predicting the probabilty of scoring within the next 5 ball actions after each interaction.")
st.write("The data utilized to develop this model is from https://www.hudl.com/blog/hudl-statsbomb-free-euro-2025-data.")

tab1, tab2 = st.tabs(["On-Ball Value Results", "On-Ball Value Model Development"])

with tab2:
    #Step 1: Understanding Match Data
    #Retrieve Match Data from StatsBombpy
    @st.cache_data(show_spinner=True)
    def get_matches (comp_id: int, season_id: int) -> pd.DataFrame: 
        m = sb.matches(competition_id = comp_id, season_id = season_id)
        m["match_id"] = m["match_id"].astype(int)
        return m 

    #Preview the Match Data
    matches = get_matches(COMPETITION_ID, SEASON_ID)
    st.subheader("1. Understanding the Match Data", divider="blue")
    st.write("First, we will import the data from Statsbombpy package, and understand what data is avaliable to us on a match-level for the 2025 Euros.")
    st.dataframe(matches.head(5), use_container_width = True)

    st.info("All matches in the Women's Euros 2025 will be utilized to train this model.")
    #Get the match IDs
    match_ids = matches["match_id"].astype(int).tolist()

    #Step 2: Pull Events for Matches
    st.subheader("2. Understanding the Event Data", divider = "blue")
    @st.cache_data(show_spinner=True)
    def load_events(match_ids):
        frames = []
        for mid in match_ids: 
            ev = sb.events(match_id=int(mid))
            frames.append(ev)
        return pd.concat(frames, ignore_index=True)

    #Load the events
    events_raw = load_events(match_ids)
    st.write("Each match has an average of 3600 events, which include but are not limited to passes, shots, and goals. We will rework and clean this data to calculate metrics and features for our model!")
    st.dataframe(events_raw.iloc[1100:1105], use_container_width=True)
    st.write("At first I thought this was empty, but realized that not all columns are being used. Every event uses a different set of columns, so we will turn our focus to the **type** column.")
    st.info("Each event is an action performed by a player on the field.")

    #Step 3: Creating functions that will help us extract data from the Events table
    def extract_xy(df: pd.DataFrame) -> pd.DataFrame: 
        
        #Create start_x/start_y and end_x/end_y from the nested lists
        def start_xy(row): 
            loc = row.get("location", None)
            if isinstance(loc, (list, tuple)) and len(loc) >=2 :
                return loc[0], loc[1]
            return np.nan, np.nan 
        
        def end_xy(row): 
            for key in ("pass_end_location", "carry_end_location", "dribble_end_location"): 
                loc = row.get(key, None)
                if isinstance(loc, (list, tuple)) and len(loc) >=2: 
                    return loc [0], loc[1]
            return np.nan, np.nan
        
        sx, sy = zip(*df.apply(start_xy, axis = 1))
        ex, ey = zip(*df.apply(end_xy, axis = 1))
        df["start_x"] = sx
        df["start_y"] = sy
        df["end_x"] = ex
        df["end_y"] = ey

        #If it is a shot, there might be a end_x/y missing, so set end to start 
        is_shot = df["type"].eq("Shot")
        df.loc[is_shot & df["end_x"].isna(), "end_x"] = df.loc[is_shot, "start_x"]
        df.loc[is_shot & df["end_y"].isna(), "end_xy"] = df.loc[is_shot, "start_y"]
        return df

    #Determine the distance to the goal 
    def dist_to_goal(x, y, goal_x = 120.0, goal_y = 40.0): 
        if not (isinstance(x, (float, int)) and isinstance (y, (float, int))): 
            if math.isnan(x) or math.isnan(y): 
                return np.nan
            else: 
                return np.nan
        
        return math.hypot(goal_x - x, goal_y - y) 

    #Step 4: Clean the data and make on the ball events table
    def prep_actions(raw: pd.DataFrame) -> pd.DataFrame:
        
        df=raw.copy()

        rename_map = {
            "type.name": "type_name",
            "team.name": "team_name",
            "player.name": "player_name",
            "possession": "possession_id",
            "shot.outcome.name": "shot_outcome",
            "pass.outcome.name": "pass_outcome",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        

        keep_types = ["Pass", "Carry", "Dribble", "Shot"]
        df = df[df["type"].isin(keep_types)].copy()

        #Make sure the rows are in match-time order 
        sort_cols = [c for c in ["match_id", "period", "minute", "second", "timestamp"] if c in df.columns]
        sort_cols = sort_cols if sort_cols else ["match_id"]
        df.sort_values(sort_cols, inplace=True)

        #Create flat XY Columns
        df=extract_xy(df)

        #Distance to the centre of the goal
        for c in ["start_x","start_y","end_x","end_y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["start_d2g"] = np.hypot(120 - df["start_x"], 40 - df["start_y"])
        df["end_d2g"]   = np.hypot(120 - df["end_x"],   40 - df["end_y"])

        #Basic flags
        df["pass_completed"] = df["pass_outcome"].isna() & df["type"].eq("Pass")
        df["shot_goal"] = df["shot_outcome"].eq("Goal")

        #The index when they have possession 
        if "possession_id" not in df.columns: 
            raise ValueError("Expected 'posession_id' columns: Check statsbompy version!")
        df["action_idx_in_poss"] = df.groupby(["match_id", "possession_id"]).cumcount()
        return df

    
    actions = prep_actions(events_raw)

    st.subheader("3. On-Ball Actions Cleaned Table", divider = "blue")
    st.write("Here we calculated where the on the field the ball starts and end with each of the actions performed. Also, we can see the distance the ball is from the centre of the opponent's net. We will use these features to train our model soon!")
    st.dataframe(actions.iloc[1100:1105][[ "match_id","team","player","type","minute","second",
        "start_x","start_y","end_x","end_y","start_d2g","end_d2g",
        "pass_completed","shot_goal","possession_id","action_idx_in_poss"]], use_container_width=True)

    #Step 5: Prepare the label, the column we want to predict, future goal in the next 5 actions (same posession)
    def label_future_goal(df: pd.DataFrame) -> pd.DataFrame: 
        df = df.copy()
        df["future_goal_5"] = 0
        gcols = ["match_id", "possession_id", "team"]
        df["_is_goal"] = df["shot_goal"].astype(int)

        def mark_group(g): 
            arr = g["_is_goal"].to_numpy()
            out = np.zeros(len(g), dtype = int)
            for i in range(len(g)): 
                j2 = min(len(g), i + 5 + 1) #Look ahead up to k actions 
                if arr[i+1:j2].sum() > 0:
                    out[i] = 1
            g["future_goal_5"] = out 
            return g
        
        return df.groupby(gcols, group_keys = False).apply(mark_group).drop(columns=["_is_goal"])

    actions = label_future_goal(actions)

    st.subheader("4. Was there be a goal in the next 5 actions?", divider = "blue")
    st.write(f"The **future_goal_5** column will be **1** if the same team scores within the next **5** actions in the **same possession**. This column will be predicted by the model later on with a probability of it being a 1 (scoring a goal within the next 5 actions).")
    st.dataframe(actions.iloc[1100:1105][[
        "match_id","team","player","type","minute","second","future_goal_5"
    ]], use_container_width=True)

    #Step 6: Build a simple state rows, before and after each action. We want to predict the on ball value before and after. 
    def build_state_rows(df: pd.DataFrame) -> pd.DataFrame: 
        #Before the Action
        before = df.copy()
        before["state_kind"] = "before"
        before["state_x"] = before["start_x"].fillna(before["start_x"].median())
        before["state_y"] = before["start_y"].fillna(before["start_y"].median())
        before["state_d2g"] = before["start_d2g"].fillna(before["start_d2g"].median())

        #After the action
        after = df.copy()
        after["state_kind"] = "after"
        after["state_x"] = after["end_x"].fillna(after["end_x"].median())
        after["state_y"] = after["end_y"].fillna(after["end_y"].median())
        after["state_d2g"] = after["end_d2g"].fillna(after["end_d2g"].median())

        cols = ["match_id","team","player","type","minute","second",
        "state_x","state_y","state_d2g","future_goal_5","state_kind"]
        states = pd.concat([before[cols], after[cols]], ignore_index=True)
        states["y"] = states["future_goal_5"].astype(int)

        return states

    states = build_state_rows(actions)

    st.subheader("5. States Table: Before and After Each Action", divider="blue")
    st.write("This table will divide each action into 2 rows. The state of the ball before the action took place, and the state of the ball after the action took place. For each state, we will predict the probability of a goal occuring 5 actions later, and then subtract the probability of the after state from the before state. The model will give us the probability of the goal occuring 5 actions later!")
    st.dataframe(states.iloc[1100:1105], use_container_width=True)

    #Step 7: Train the Model
    def train_value_model(states: pd.DataFrame): 
        num_cols = ["state_x","state_y","state_d2g","minute","second"]
        cat_cols = ["type"]

        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])

        model = Pipeline([
            ("pre", pre), #transform the features
            ("lr", LogisticRegression(max_iter=200))
        ])

        X = states[num_cols+cat_cols]
        y = states["y"]
        model.fit(X, y)

        auc = roc_auc_score(y, model.predict_proba(X)[:,1]) if y.nunique() > 1 else float("nan")
        
        return model, (num_cols+cat_cols), auc
    
    model, feat_cols, auc = train_value_model(states)

    st.subheader("6. Training and Evaluating the Model", divider="blue")
    st.write("The model we utilized was a Logistic Regression model. This model utilizes the features to predict the probability of the label being a 1 (scoring a goal within the next 5 actions) or 0. If the probability is above 0.5, then it will predict 1 (goal happening soon!). ")
    st.write("AUC (Area under the ROC Curve) measures the probability of that our model ranks a random positive example higher than a random negative examples. If your AUC is perfect, it means there is perfect separation and the model is almost always correct! If it is under 0.5 or is 0.5, then it is guessing randomly. Thus, we are determining the quality of our model. ")
    st.markdown(f":blue-background[Train AUC:] **{auc:3f}**")
    st.caption("The model needs to be further evaluated, however the goal is to develop a simple model to assess the Euros.")
    st.write("This is pretty good considering the simplification of our features and using a simple logistic model.")
    

    #Step 8: Compute On Ball Value (Action) = V(After) - V(Before)
    def compute_obv(df_actions: pd.DataFrame, model, feat_cols): 
        st_states = build_state_rows(df_actions).reset_index(drop=True)
        X = st_states[feat_cols]
        st_states["V"] = model.predict_proba(X)[:,1].astype(float)

        #Each action has a before and after state, so ensure it is repeated with the same indices of the action 
        actions = df_actions.reset_index(drop=True).copy()
        actions["action_row_id"] = actions.index

        #Two rows per action
        n = len(actions)
        ids_before = np.arange(n)
        ids_after  = np.arange(n)
        st_states["action_row_id"] = np.r_[ids_before, ids_after]

        #Ensuring that each action its V_before and after values side by side, instead of two seperate entries  
        V_wide = (
            st_states
            .pivot_table(index="action_row_id", columns="state_kind", values="V", aggfunc="first")
            .reset_index()
            .rename(columns={"before": "V_before", "after": "V_after"})
        )

        actions = actions.drop(columns=["V_before","V_after","OBV"], errors="ignore")

        #Add the columns to the rows
        out = actions.merge(V_wide, on="action_row_id", how="left", validate="one_to_one")
        out["V_before"] = pd.to_numeric(out["V_before"], errors="coerce")
        out["V_after"]  = pd.to_numeric(out["V_after"],  errors="coerce")
        out["OBV"] = out["V_after"] - out["V_before"]

        return out
       
    
    obv_actions = compute_obv(actions, model, feat_cols)
    st.subheader("7. Calcuating the On-Ball Value of Each Action", divider="blue")
    st.markdown("We apply the model to our data to calculate the probability of scoring before the action and after the action. Thereafter calculate the following: :blue-background[Probability of Scoring After the Action - Probability of Scoring Before the Action = On-Ball Value].")
    st.dataframe(obv_actions.head()[[
        "match_id","team","player","type","minute","second","V_before","V_after","OBV"
    ]], use_container_width=True)

    #Step 9: Develop a payer leaderboard, to determine which player had the best OBV value. 
    player_obv = (obv_actions.groupby(["player", "team"], dropna = False)["OBV"].sum().reset_index().sort_values("OBV", ascending=False).head(10))

    st.subheader("8. Sorting the Table to Determine the Top Players by their On-Ball Values", divider = "rainbow")
    st.write("We can rank the players by their cumulative on-ball value througout the tournament to determine which players added the most value to their team and help their team score goals!")
    st.dataframe(player_obv, use_container_width=True)
    st.success("Further analysis with the predicted OBV is on the results page!")

# --- helpers (top-level, outside the tab) --- #Using ChatGpt to improve the performance
@st.cache_data(show_spinner=False)
def teams_list(oa: pd.DataFrame):
    # if already categorical, this is O(k); otherwise still fast and cached
    s = oa["team"].dropna()
    if hasattr(s.dtype, "name") and "category" in s.dtype.name:
        return sorted(s.cat.categories.tolist())
    return sorted(s.astype(str).unique().tolist())

@st.cache_data(show_spinner=False)
def players_for_team(oa: pd.DataFrame, team: str):
    s = oa.loc[oa["team"] == team, "player"].dropna()
    return sorted(s.astype(str).unique().tolist())

@st.cache_data(show_spinner=False)
def topk_player_counts(oa: pd.DataFrame, top_pairs: pd.DataFrame):
    # compute actions/total_obv only for the top visible players
    key = tuple(map(tuple, top_pairs[["player","team"]].values))
    sub = oa.merge(top_pairs[["player","team"]].drop_duplicates(), on=["player","team"], how="inner")
    g = (sub.groupby(["player","team"], dropna=False)["OBV"]
            .agg(actions="size", total_obv="sum").reset_index())
    return g

def _progress_col(label="OBV bar"):
    # nicer in-cell bar without Styler (lighter memory)
    return st.column_config.ProgressColumn(
        label, min_value=0.0, max_value=None, format="%.3f"
    )

# ---------------- TAB 1 (drop-in) ----------------
with tab1:
    st.subheader("🏆 Overall 2025 Euros Analysis")

    # === Top 10 players table ===
    # keep this tiny; don't compute counts for all players
    top10 = (
        player_obv.reset_index(drop=True).copy()
        .sort_values("OBV", ascending=False)
        .head(10)
    )

    # medal/rank column (vectorized)
    r = np.arange(1, len(top10) + 1)
    medal = np.array([""] * len(top10), dtype=object)
    if len(top10) >= 1: medal[0] = "🥇"
    if len(top10) >= 2: medal[1] = "🥈"
    if len(top10) >= 3: medal[2] = "🥉"
    top10.insert(0, "#", np.where(medal == "", np.char.zfill(r.astype(str), 2), medal))

    # If we have obv_actions, compute counts ONLY for these 10
    if obv_actions is not None and len(top10):
        counts10 = topk_player_counts(obv_actions, top10[["player","team"]])
        tbl = (top10.drop(columns=["OBV"])
                    .merge(counts10.rename(columns={"total_obv":"OBV"}), on=["player","team"], how="left"))
        tbl["avg_obv"] = tbl["OBV"] / tbl["actions"].replace(0, np.nan)
    else:
        tbl = top10.copy()
        tbl["actions"] = np.nan
        tbl["avg_obv"] = np.nan

    tbl = tbl[["#", "player", "team", "OBV", "actions", "avg_obv"]]
    # Display without Styler to avoid large HTML; use column_config instead
    st.markdown("#### Top 10 Players Ranked by OBV")
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.TextColumn("#", width="small"),
            "player": st.column_config.TextColumn("Player"),
            "team": st.column_config.TextColumn("Team"),
            "OBV": _progress_col("OBV"),
            "actions": st.column_config.NumberColumn("Actions", format="%.0f"),
            "avg_obv": st.column_config.NumberColumn("avg OBV", format="%.4f"),
        }
    )
    st.caption("OBV = change in P(goal within the next 5 actions). ‘avg_obv’ ≈ efficiency per action.")

    st.write("You might be wondering why isn't Spain's Esther González, the top scorer at Euros 2025, in the top 10 for the players that added the most on-ball value.")
    st.write("This model highly rewards players who often move balls into threatening states through long progressive passes/carries, which means that the difference between the probabilty of scoring before their action and after their action is large, hence their OBV is higher. Strikers who are required to perform clincal finishes to score a goal would have a smaller difference, as the ball might already be in a threatening location.")
    st.write("Our model can be improved by including more features, better understanding which features impact the model the most, and using a more advanced machine learning tool such as XGBoost.")

    # === Team rankings table (per-team aggregate only) ===
    st.markdown("#### Players in a Team Ranked by OBV")
    col1, col2 = st.columns([0.25, 0.75])

    with col1:
        teams = teams_list(obv_actions if obv_actions is not None else pd.DataFrame(columns=["team"]))
        team_pick = st.selectbox("Select a team", options=teams, index=0 if teams else None)

    with col2:
        if obv_actions is not None and team_pick:
            team_df = obv_actions.loc[obv_actions["team"] == team_pick, ["player","team","OBV"]]
            leader = (team_df.groupby(["player","team"], dropna=False)["OBV"]
                              .agg(OBV="sum", actions="size")
                              .reset_index()
                              .sort_values("OBV", ascending=False)
                              .reset_index(drop=True))
            leader["avg_obv"] = leader["OBV"] / leader["actions"].replace(0, np.nan)
            leader.insert(0, "#", np.char.zfill((np.arange(1, len(leader)+1)).astype(str), 2))
            # medals on top 3
            for i, medal_ in enumerate(["🥇","🥈","🥉"]):
                if i < len(leader): leader.at[i, "#"] = medal_
            team_tbl = leader[["#", "player", "team", "OBV", "actions", "avg_obv"]]
        else:
            team_tbl = pd.DataFrame(columns=["#", "player", "team", "OBV", "actions", "avg_obv"])

        st.markdown(f"##### {team_pick if team_pick else '—'} — OBV Rankings")
        st.dataframe(
            team_tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.TextColumn("#", width="small"),
                "player": st.column_config.TextColumn("Player"),
                "team": st.column_config.TextColumn("Team"),
                "OBV": _progress_col("OBV"),
                "actions": st.column_config.NumberColumn("Actions", format="%.0f"),
                "avg_obv": st.column_config.NumberColumn("avg OBV", format="%.4f"),
            }
        )
        st.caption("OBV = change in P(goal within 5 actions). ‘avg_obv’ ≈ efficiency per action. Actions = on-ball events counted.")

    st.write("As seen above, that the number of actions can be correlated to the a player's OBV. Hence next steps would be to calculate the OBV per 90 minutes, as not all players get the same amount of playing time. Thus, this model and the results heavily favours players that get more playing time.")
    st.write("Also as mentioned before, a lot of the strikers that were critical to the game are lower on the list. How can we tune the model to remove this bias?")

    # === Player progression line ===
    st.markdown("#### Players's OBV Progression Throughout the Tournament")
    col11, col22 = st.columns([0.25, 0.75])

    with col11:
        pick_team = st.selectbox("Pick a team", options=teams, index=0 if teams else None, key="pick_team2")
        players = players_for_team(obv_actions, pick_team) if (obv_actions is not None and pick_team) else []
        player_pick = st.selectbox("Pick a player", options=players, index=0 if players else None)

    with col22:
        if obv_actions is not None and pick_team and player_pick:
            p_df = obv_actions.loc[(obv_actions["team"] == pick_team) & (obv_actions["player"] == player_pick), ["match_id","OBV"]]
            per_match = (p_df.groupby("match_id", dropna=False)["OBV"].sum()
                            .reset_index().rename(columns={"OBV":"obv_match"}))

            # Optional match meta (kept slim to only needed cols)
            if isinstance(matches, pd.DataFrame) and {"match_id","match_date","home_team","away_team"}.issubset(matches.columns):
                meta = matches[["match_id","match_date","home_team","away_team"]].copy()
                per_match = per_match.merge(meta, on="match_id", how="left")
                per_match["match_date"] = pd.to_datetime(per_match["match_date"], errors="coerce")
                per_match["opponent"] = np.where(per_match["home_team"] == pick_team, per_match["away_team"], per_match["home_team"])
                per_match = per_match.sort_values(["match_date","match_id"])
                x_axis = per_match["match_date"]; x_label = "Match Date"
            else:
                per_match = per_match.sort_values("match_id")
                x_axis = per_match["match_id"]; x_label = "Match ID"

            fig = px.line(
                per_match,
                x=x_axis, y="obv_match",
                markers=True,
                hover_data={"match_id": True, "obv_match": ":.3f", "opponent": True} if "opponent" in per_match else {"match_id": True, "obv_match": ":.3f"},
                labels={"obv_match": "OBV (per match)", x_axis.name if hasattr(x_axis, "name") else "x": x_label},
                title=f"{player_pick} — OBV by Match ({pick_team})"
            )
            fig.update_traces(mode="lines+markers", line=dict(width=3))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(size=14), margin=dict(l=10,r=10,t=50,b=10), hovermode="x unified")
            fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="rgba(150,150,150,0.4)")
            fig.update_xaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)
            del fig  # free reference
        st.caption("Hover to see opponent and exact OBV values.")

# (Optional) clean up big temporaries created in this block
import gc
del top10, tbl, team_tbl
gc.collect()

# Footer stays the same
st.caption("All credit for creating an OBV model goes to Hudl Statsbomb. As well as providing free open access data for the 2025 Euros. AI was utilized to improve styling, code performance and general development. All commentary and thoughts are my own. Please free to connect with me [LinkedIn](https://www.linkedin.com/in/eshah17) to share feedback or have discussion. Thank you!")
