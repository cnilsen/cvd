import numpy
from matplotlib import pyplot
import pandas

import ipywidgets
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


def _maybe_cumsum(df, grouplevels, doit):
    if doit:
        return df.groupby(level=grouplevels).cumsum()
    return df


def _maybe_percapita(df, doit):
    if doit:
        return (
            df.divide(_pop_by_country, axis='index', level='Country/Region')
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


def global_data():
    return (
        _global_data('confirmed')
            .join(_global_data('deaths'))
            .groupby(level=['Country/Region', 'Date'])
            .sum()
            .groupby(level='Country/Region')
            .apply(
                lambda g: g.diff()
                           .where(lambda x: x.notnull(), g)
                           .where(lambda x: x >= 0, 0) 
            )
            .reset_index()
            .assign(Subset=lambda df: numpy.select(
                (df['Country/Region'].isin(('US', 'Canada')), df['Country/Region'] == 'China'),
                ('US/Canada', 'China'), 'Everywhere Else'
            ))
    )


def load_data():
    return pandas.concat(
        map(_read_data, ['Confirmed', 'Recovered', 'Deaths']),
        axis='columns'
    ).reset_index().assign(Subset=lambda df: numpy.select(
        (df['Country/Region'].isin(('US', 'Canada')), df['Country/Region'] == 'China'),
        ('US/Canada', 'China'), 'Everywhere\nElse'
    ))


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


def _state_data(us_data, state, dates, cumulative=True):
    start_date, end_date = ('2020-' + d for d in dates)
    return (
        us_data
            .groupby(['State', 'Date'])
            .sum()
            .xs(state, level='State', axis='index')
            .pipe(_maybe_cumsum, cumulative)
            .loc[start_date:end_date]
            .rename_axis(columns='Metric')
            .stack()
            .to_frame('Cases')
            .reset_index()
            .assign(Metric=lambda df: pandas.Categorical(
                df['Metric'], categories=['Confirmed', 'Recovered', 'Deaths'], ordered=True)
            )
    )


def _us_data(us_data, dates, cumulative=True):
    start_date, end_date = ('2020-' + d for d in dates)
    return (
        us_data.groupby(by=['Date'])
            .sum()
            .pipe(_maybe_cumsum, cumulative)
            .loc[start_date:end_date]
            .reset_index()
            .melt(id_vars=['Date'], var_name='Metric', value_name='Cases')
    )


def _area_chart(tidy_data):
    palette = altair.Scale(scheme='accent')
    conf_chart = (
        altair.Chart(tidy_data, width=400, height=250)
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

    return (conf_chart + outcome_chart)


def us_chart(us_data):
    dates = us_data['Date'].unique()
    options = [pandas.Timestamp(d).strftime('%m/%d') for d in dates]
    df = ipywidgets.fixed(us_data)
    state_dropdown = ipywidgets.Dropdown(
        options=_state_abbrev.items(),
        value='US',
        description='Pick a State:',
        disabled=False
    )
    date_slider = ipywidgets.SelectionRangeSlider(
        index=(0, len(dates)-1),
        options=options,
        description='Dates',
        disabled=False,
        layout=ipywidgets.Layout(width='500px'),
        continuous_update=False
    )
    cumulative_toggle = ipywidgets.RadioButtons(
        options=['Cumulative', 'NEW'],
        description='Cumulative or NEW?',
        disabled=False
    )

    def chart(us_data, state, cumulative, dates):
        _doit = cumulative.lower() == 'cumulative'
        if state == 'US':
            data = _us_data(us_data, dates, _doit)
        else:
            data = _state_data(us_data, state, dates, _doit)
        return _area_chart(data)

    return ipywidgets.interact(
        chart,
        us_data=df,
        state=state_dropdown,
        cumulative=cumulative_toggle,
        dates=date_slider
    )


def _global_ts_chart(data, how, dates):
    start, end = [pandas.Timestamp('2020-' + d) for d in dates]
    _data = (
        data.groupby(by=['Subset', 'Date'],)
            .sum()
            .pipe(_maybe_cumsum, 'Subset', (how.lower() == 'cumulative'))
            .unstack(level='Subset')
            .loc[start:end]
            .stack(level='Subset')
    )

    colors_scale = altair.Scale(
        domain=['China', 'US/Canada', 'Everywhere Else', 'Global Deaths'],
        range=['Crimson', 'SteelBlue', 'SaddleBrown', 'DimGrey']
    )

    conf = (
        _data.reset_index()
            .pipe(altair.Chart, width=400, height=250)
            .mark_area()
            .encode(
                x=altair.X('Date', type='temporal'),
                y=altair.Y('Confirmed', type='quantitative', stack=True),
                color=altair.Color('Subset', scale=colors_scale),
                order=altair.Order('Subset', sort='ascending'),
            )
    )

    dead = (
        _data.groupby(level='Date').sum()
            .reset_index()
            .assign(Subset='Global Deaths')
            .pipe(altair.Chart, width=400, height=250)
            .mark_area()
            .encode(
                x=altair.X('Date', type='temporal'),
                y=altair.Y('Deaths', type='quantitative', stack=True),
                color=altair.Color('Subset', scale=colors_scale),
            )
    )
    
    return (conf + dead)


def global_ts_chart(data):
    dates = data['Date'].unique()
    options = [pandas.Timestamp(d).strftime('%m/%d') for d in dates]
    df = ipywidgets.fixed(data)
    date_slider = ipywidgets.SelectionRangeSlider(
        index=(0, len(dates)-1),
        options=options,
        description='Dates',
        disabled=False,
        layout=ipywidgets.Layout(width='500px'),
        continuous_update=False
    )
    cumulative_toggle = ipywidgets.RadioButtons(
        options=['Cumulative', 'NEW'],
        description='How?',
        disabled=False
    )

    return ipywidgets.interact(
        _global_ts_chart,
        data=df,
        how=cumulative_toggle,
        dates=date_slider
    )


def _new_cases_chart(data, how, which, N, percapita):
    countries = (
        data.groupby('Country/Region')
            ['Confirmed']
            .sum()
            .sort_values(ascending=False)
            .head(N)
            .index.tolist()
    )

    if N <= 10:
        palette = altair.Scale(scheme='category10')
    else:
        palette = altair.Scale(scheme='category20')

    return (
        data.loc[data['Country/Region'].isin(countries)]
            .pipe(new_cases_since_nth, 'Country/Region', 100)
            .pipe(_maybe_percapita, percapita)
            .reset_index()
            .pipe(altair.Chart, width=400, height=250)
            .mark_line()
            .encode(
                x=altair.X('days_since', type='quantitative'),
                y=altair.Y(how.title() + '_' + which.title(), type='quantitative'),
                color=altair.Color('Country/Region', scale=palette)
            )
    )


def new_cases_chart(data):
    df = ipywidgets.fixed(data)
    how_toggle = ipywidgets.RadioButtons(
        options=['Cumulative', 'NEW'],
        description='How?',
        disabled=False
    )
    which_toggle = ipywidgets.RadioButtons(
        options=['Confirmed', 'Deaths'],
        description='Which?',
        disabled=False,
        orientation='horizontal'
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

    percap_toggle = ipywidgets.RadioButtons(
        options=[True, False],
        description='Per 100k?',
        disabled=False,
        orientation='horizontal'
    )

    return ipywidgets.interact(
        _new_cases_chart,
        data=df,
        how=how_toggle,
        which=which_toggle,
        N=N_slider,
        percapita=percap_toggle
    )