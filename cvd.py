from copy import deepcopy

import numpy
from matplotlib import pyplot
import pandas

import ipywidgets
import altair


_pop_by_region = pandas.Series({
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
    'Alabama': 4903185,
    'Alaska': 731545,
    'Arizona': 7278717,
    'Arkansas': 3017804,
    'California': 39512223,
    'Colorado': 5758736,
    'Connecticut': 3565287,
    'Delaware': 973764,
    'District of Columbia': 705749,
    'Florida': 21477737,
    'Georgia': 10617423,
    'Hawaii': 1415872,
    'Idaho': 1787065,
    'Illinois': 12671821,
    'Indiana': 6732219,
    'Iowa': 3155070,
    'Kansas': 2913314,
    'Kentucky': 4467673,
    'Louisiana': 4648794,
    'Maine': 1344212,
    'Maryland': 6045680,
    'Massachusetts': 6892503,
    'Michigan': 9986857,
    'Minnesota': 5639632,
    'Mississippi': 2976149,
    'Missouri': 6137428,
    'Montana': 1068778,
    'Nebraska': 1934408,
    'Nevada': 3080156,
    'New Hampshire': 1359711,
    'New Jersey': 8882190,
    'New Mexico': 2096829,
    'New York': 19453561,
    'North Carolina': 10488084,
    'North Dakota': 762062,
    'Ohio': 11689100,
    'Oklahoma': 3956971,
    'Oregon': 4217737,
    'Pennsylvania': 12801989,
    'Rhode Island': 1059361,
    'South Carolina': 5148714,
    'South Dakota': 884659,
    'Tennessee': 6829174,
    'Texas': 28995881,
    'Utah': 3205958,
    'Vermont': 623989,
    'Virginia': 8535519,
    'Washington': 7614893,
    'West Virginia': 1792147,
    'Wisconsin': 5822434,
    'Wyoming': 578759,
    'Puerto Rico': 3193694,
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


def _make_cumulative_toggle():
    return ipywidgets.ToggleButtons(
        options=['Cumulative', 'NEW'],
        description='How?',
        disabled=False,
        button_style='',
        tooltips=['Cumulative counts', 'New counts each day']
    )


def _make_date_slider(dates):
    options = [pandas.Timestamp(d).strftime('%m/%d') for d in dates]
    return ipywidgets.SelectionRangeSlider(
        index=(0, len(dates)-1),
        options=options,
        description='Dates',
        disabled=False,
        layout=ipywidgets.Layout(width='400px'),
        continuous_update=False
    )


def _make_region_selector(data, regioncol, *highlights):
    _regions = (
        data.groupby([regioncol])
            ['Confirmed'].sum()
            .sort_values(ascending=False)
            .index.to_list()
    )
    regions = [*highlights, *[s for s in _regions if s not in highlights]]
    return ipywidgets.SelectMultiple(
        options=regions,
        value=highlights,
        rows=10,
        description=regioncol,
        disabled=False
    )


def _maybe_cumsum(df, grouplevels, doit):
    if doit:
        return df.groupby(level=grouplevels).cumsum()
    return df


def _maybe_percapita(df, doit, regioncol):
    if doit:
        return (
            df.divide(_pop_by_region, axis='index', level=regioncol)
              .multiply(100000)
        )
    else:
        return df


def _global_data(part):
    url = (
        "https://raw.githubusercontent.com/CSSEGISandData/"
        "COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
        "time_series_covid19_{}_global.csv"
    )
    return (
        pandas.read_csv(url.format(part.lower()))
            .fillna({'Province/State': 'default'})
            .drop(columns=['Lat', 'Long'])
            .set_index(['Province/State', 'Country/Region'])
            .rename(columns=pandas.to_datetime)
            .rename_axis(columns=['Date'])
            .stack().to_frame(part.title())
    )


def _unaccumulate(df, level):
    return (
        df.groupby(level=level)
          .apply(
              lambda g: g.diff()
                         .where(lambda x: x.notnull(), g)
                         .where(lambda x: x >= 0, 0)
          )
    )


def _break_out_region(df, regioncol, default, *regions):
    return (
        pandas.concat([
            df.loc[df[regioncol].isin(regions)].assign(Subset=df[regioncol]),
            df.assign(Subset=default)
        ], sort=False, ignore_index=True)
        .groupby(by=['Date', 'Subset'])
        .sum()
        .fillna(0).astype(int)
        .sort_index()
    )


def global_data():
    return (
        _global_data('confirmed')
            .join(_global_data('deaths'))
            .groupby(level=['Country/Region', 'Date'])
            .sum()
            .pipe(_unaccumulate, level='Country/Region')
            .reset_index()
            .rename(columns={'Country/Region': 'Country'})
    )


def state_data():
    url = (
        "https://raw.githubusercontent.com/"
        "nytimes/covid-19-data/master/us-states.csv"
    )
    return (
        pandas.read_csv(url, parse_dates=['date'])
            .rename(columns=str.title)
            .rename(columns={'Cases': 'Confirmed'})
            .drop(columns=['Fips'])
            .set_index(['Date', 'State'])
            .sort_index()
            .pipe(_unaccumulate, level='State')
            .astype(int)
            .reset_index()
    )


def new_cases_since_nth(data, regioncol, nth, metric='Confirmed', datecol='Date'):
    since = (
        data.groupby(by=[regioncol, datecol])
                .sum()
            .rename(columns=lambda c: 'New_'+ c)
            .groupby(by=[regioncol])
                .apply(lambda g: g.assign(
                    Cumulative_Confirmed=g['New_Confirmed'].cumsum(),
                    Cumulative_Deaths=g['New_Deaths'].cumsum(),
                    days_since=g['New_' + metric].cumsum().ge(nth).cumsum())
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


def _new_cases_chart(data, how, which, percapita, N, regioncol):
    countries = (
        data.groupby(regioncol)
            ['Confirmed']
            .sum()
            .sort_values(ascending=False)
            .head(N)
            .index.tolist()
    )

    if how.lower() != 'new':
        yscale = altair.Scale(type='log') 
    else:
        yscale = altair.Scale(type='linear') 

    if N <= 10:
        palette = altair.Scale(scheme='category10')
    else:
        palette = altair.Scale(scheme='category20')

    return (
        data.loc[data[regioncol].isin(countries)]
            .pipe(new_cases_since_nth, regioncol, 100)
            .pipe(_maybe_percapita, percapita, regioncol)
            .reset_index()
            .pipe(altair.Chart, width=400, height=250)
            .mark_line()
            .encode(
                x=altair.X('days_since', type='quantitative'),
                y=altair.Y(
                    how.title() + '_' + which.title(),
                    type='quantitative',
                    scale=yscale
                ),
                color=altair.Color(regioncol, scale=palette)
            )
    )


def new_cases_chart(data, regioncol):
    df = ipywidgets.fixed(data)


    which_toggle = ipywidgets.ToggleButtons(
        options=['Confirmed', 'Deaths'],
        description='Which?',
        disabled=False,
        button_style='',
        tooltips=['Confirmed COVID Cases', 'COVID-related deaths']
    )

    N_slider = ipywidgets.IntSlider(
        value=10,
        min=4,
        max=20,
        step=1,
        description='Top N:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )

    percap_toggle = ipywidgets.ToggleButtons(
        options=[True, False],
        description='Per 100k?',
        disabled=False,
        button_style='',
        tooltips=['Incidents per 100k', 'Absolute totals']
    )

    return ipywidgets.interact(
        _new_cases_chart,
        data=df,
        how=_make_cumulative_toggle(),
        which=which_toggle,
        percapita=percap_toggle,
        N=N_slider,
        regioncol=ipywidgets.fixed(regioncol)
    )


def _break_out_chart(data, cumulative, regioncol, default, regions, dates):
    start, end = [pandas.Timestamp('2020-' + d) for d in dates]
    df = (
        data
        .pipe(_break_out_region, regioncol, default, *regions)
        .pipe(_maybe_cumsum, ['Subset'], cumulative != 'NEW')
        .loc[(slice(start, end), slice(None))]
        .reset_index()
    )

    whole = (
        altair.Chart(df)
            .properties(width=500, height=300)
            .mark_area()
            .encode(
                x=altair.X('Date', type='temporal'),
                y=altair.Y('Confirmed', type='quantitative', stack=True),
                color=altair.value("lightgrey")
            )
            .transform_filter(altair.datum.Subset == default)
    )

    parts = (
        altair.Chart(df)
            .properties(width=500, height=300)
            .mark_area()
            .encode(
               x=altair.X('Date', type='temporal'),
               y=altair.Y('Confirmed', type='quantitative', stack=True),
               color=altair.Color('Subset', type='nominal'),
            )
            .transform_filter(altair.datum.Subset != default)
    )

    return whole + parts


def break_out_US(data):
    return ipywidgets.interact(
        _break_out_chart,
        data=ipywidgets.fixed(data),
        cumulative=_make_cumulative_toggle(),
        regioncol=ipywidgets.fixed('State'),
        default=ipywidgets.fixed('US'),
        regions=_make_region_selector(data, 'State', 'Oregon', 'Washington'),
        dates=_make_date_slider(data['Date'].unique())
    )


def break_out_world(data):
    return ipywidgets.interact(
        _break_out_chart,
        data=ipywidgets.fixed(data),
        cumulative=_make_cumulative_toggle(),
        regioncol=ipywidgets.fixed('Country'),
        default=ipywidgets.fixed('World'),
        regions=_make_region_selector(data, 'Country', 'US', 'Italy', 'Spaid'),
        dates=_make_date_slider(data['Date'].unique())
    )
