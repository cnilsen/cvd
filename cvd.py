import numpy
from matplotlib import pyplot
import pandas

import altair


_pop_by_country = pandas.Series({
    'Austria': 8.882e6,
    'Belgium': 11.4e6,
    'China': 1386e6,
    'France': 66.99e6,
    'Germany': 82.79e6,
    'Iran': 81.16e6,
    'Italy': 60.48e6,
    'Korea, South': 51.47e6,
    'Netherlands': 17.18e6,
    'Norway': 5.368e6,
    'Spain': 44.66e6,
    'Sweden': 10.12e6,
    'Switzerland': 8.57e6,
    'US': 372.2e6,
    'United Kingdom': 66.44e6,
}, name='Population')


_state_abbrev = {
    'United States': 'US',
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'Washington, D.C.': 'WashDC',
    'District of Columbia': 'WashDC',
    'Grand Princess': 'Cruise Ships',
    'Diamond Princess': 'Cruise Ships',
}

_abbrev_state = {value: key for key, value in _state_abbrev.items()}


def read_data(tspart):
    url = (
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
        "csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_19-covid-{}.csv"
    ).format(tspart)
    return (
        pandas.read_csv(url)
        .drop(columns=['Lat', 'Long'])
        .fillna({'Province/State': 'default'})
        .set_index(['Country/Region', 'Province/State'])
        .rename(columns=pandas.to_datetime)
        .sort_index(axis='columns')
        .pipe(lambda df: df.diff(axis='columns')
                           .where(lambda x: x.notnull(), df)
                           .where(lambda x: x >= 0, 0)
        )
        .rename_axis(columns=['Date'])
        .astype(int)
        .stack(level='Date')
        .to_frame(tspart)
    )


@pyplot.FuncFormatter
def yaxisfmt(tick, pos):
    thousands = int(tick // 1000)
    return f'{thousands}k'


def cumulative_xtab(data, regioncol, datecol='Date'):
    return (
    data.groupby(by=[regioncol, datecol]).sum()
        .unstack(level=regioncol)
        .cumsum()
        .rename_axis(columns=['Metric', regioncol])
)


def cumulative_plot_global(cumxtab):
    cases = (
        cumxtab
            .loc[:, 'Confirmed']
            .pipe(lambda df:
                df.rename(columns=lambda c: f"{c} ({df.loc[:, c].iloc[-1]:,d})")
            )   
    )

    outcomes = (
        cumxtab
            .groupby(level='Metric', axis='columns').sum()
            .loc[:, ['Deaths', 'Recovered']]
            .pipe(lambda df:
                df.rename(columns=lambda c: f"Global {c} ({df.loc[:, c].iloc[-1]:,d})")
            ) 
    )

    fig1, ax1 = pyplot.subplots(figsize=(8, 5), nrows=1, sharex=True)
    cases.plot.area(ax=ax1, alpha=0.875, zorder=0, linewidth=0)
    outcomes.plot.area(ax=ax1, zorder=10, color=['0.75', '0.25'], linewidth=0, alpha=0.5)

    ax1.set_ylabel('Total Cases')
    ax1.set_xlabel('')

    ax1.yaxis.set_major_formatter(yaxisfmt)

    return fig1


def new_cases_since_nth(data, regioncol, nth, metric='Confirmed', datecol='Date'):
    since = (
        data.groupby(by=[regioncol, datecol])
                .sum()
            .groupby(by=[regioncol])
                .apply(lambda g: g.assign(
                    cmlcase=g[metric].cumsum(),
                    days_since=g[metric].cumsum().ge(nth).cumsum())
                )
            .rename_axis(columns='Metric')
            .loc[lambda df: df['days_since'] >= 1]
            .set_index('days_since', append=True)
            .reset_index(datecol, drop=True)
    )
    return since


def days_until_nth_accumulation(since, regioncol, nth, accumcol='cmlcase'):
    return (
        since
            .loc[lambda df: df[accumcol] > 10000]
            .reset_index()
            .groupby(by=[regioncol])
            .first()
            .sort_values(by=['days_since'])
            .loc[:, 'days_since']
    )


def get_us_state_data(data):
    return (
        data.loc[data['Country/Region'] == 'US']
            .replace({'Province/State': _state_abbrev})
            .assign(State=lambda df: df['Province/State'].where(
                ~df['Province/State'].str.contains(', '),
                df['Province/State'].str.split(', ').str[-1]
            ))
            .drop(columns=['Country/Region', 'Subset'])
    )

def plot_us_state_data(us_data, state, ax=None, **kwargs):
    return (
        us_data.groupby(by=['State', 'Date'])
          .sum()
          .xs(state, level='State')
          .cumsum()
          .loc[lambda df: df['Confirmed'] > 1]
          .plot.area(ax=ax, **kwargs)  
    )


def _state_data(us_data, state, dates):
    start_date, end_date = ('2020-' + d for d in dates)
    return (
        us_data
            .groupby(['State', 'Date'])
            .sum()
            .xs(state, level='State', axis='index')
            .cumsum()
            .loc[start_date:end_date]
            .rename_axis(columns='Metric')
            .stack()
            .to_frame('Cases')
            .reset_index()
            .assign(Metric=lambda df: pandas.Categorical(
                df['Metric'], categories=['Confirmed', 'Recovered', 'Deaths'], ordered=True)
            )
    )


def _us_data(us_data, dates):
    start_date, end_date = ('2020-' + d for d in dates)
    return (
        us_data.groupby(by=['Date'])
            .sum()
            .cumsum()
            .loc[start_date:end_date]
            .reset_index()
            .melt(id_vars=['Date'], var_name='Metric', value_name='Cases')
    )


def _area_chart(tidy_data):
    palette = altair.Scale(scheme='category10')   
    conf_chart = (
        altair.Chart(tidy_data, width=400, height=175)
            .mark_area()
            .encode(
                x=altair.X('Date', type='temporal'),
                y=altair.Y('Cases', type='quantitative', stack=True),
                color=altair.Color('Metric', scale=palette)
            )
            .transform_filter(
                altair.datum.Metric == 'Confirmed'
            )
    )

    outcome_chart = (
        altair.Chart(tidy_data, width=400, height=175)
            .mark_area()
            .encode(
                x=altair.X('Date', type='temporal'),
                y=altair.Y('Cases', type='quantitative', stack=True),
                color=altair.Color('Metric', scale=palette),
                order=altair.Order('Metric', sort="ascending")
            )
            .transform_filter(
                altair.datum.Metric != 'Confirmed'
            )
    )

    return (conf_chart + outcome_chart).configure_mark(opacity=0.875)