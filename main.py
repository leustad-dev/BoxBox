import fastf1

session = fastf1.get_session(2025, 23, "Q")
event_schedule = fastf1.get_event_schedule(2025, include_testing=True)
print(session.name, session.date, session.event["EventName"])
# print(event_schedule['RoundNumber'].astype(str) + " " + event_schedule['Country'])
print(event_schedule.columns)

# session.load()
# # print(session.results)
#
# results = session.results
# print(results[['Abbreviation', 'Position', 'Time']])
#
# laps = session.laps
# # Example: last lap for each driver
# latest_laps = laps.groupby('Driver').last()
# print(latest_laps.columns)
#
# cols = [c for c in ['LapTime', 'Position', 'GapToLeader', 'Interval'] if c in latest_laps.columns]
# print(latest_laps[cols])
