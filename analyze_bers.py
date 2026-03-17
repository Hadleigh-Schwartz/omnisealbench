import pandas as pd

# load the CSV file into a DataFrame
df = pd.read_csv("final_unattacked_bers.csv")
# filter out any entries that contain "group_office"
df = df[~df["Recording Path"].str.contains("group_office")]

# get average timbre ber
timbre_df = df[(df["Model"] == "timbre")]
audioseal_df = df[(df["Model"] == "audioseal")]

avg_timbre_ber = timbre_df["BER"].mean()
avg_audioseal_ber = audioseal_df["BER"].mean()

print(f"Average Timbre BER: {avg_timbre_ber}")
print(f"Average AudioSeal BER: {avg_audioseal_ber}")