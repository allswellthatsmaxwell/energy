library(glue)
library(ggplot2)
library(magrittr)

DATA_DIR <- '../data'

data_file <- function(fname) {
    glue::glue("{DATA_DIR}/{fname}")
}

files <- list(
    meter = list(train = data_file('train.csv'),
                 test = data_file('test.csv')),
    weather = list(train = data_file('weather_train.csv'),
                   test = data_file('weather_test.csv')),
    buildings = data_file('building_metadata.csv'))

buildings <- readr::read_csv(files$buildings)
meter <- readr::read_csv(files$meter$train)
weather <- readr::read_csv(files$weather$train) %>%
  dplyr::mutate(site_id = factor(site_id))

weather %<>% 
  dplyr::mutate(day = as.integer(format(timestamp, "%j")),
                week = as.integer(format(timestamp, "%U")))

buildings_per_site <- buildings %>%
  dplyr::group_by(site_id) %>%
  dplyr::summarize(dplyr::n())
buildings_per_site

## Rows in weather are by the minute, not the day.
daily_averages <- weather %>%
  dplyr::group_by(site_id, day) %>%
  dplyr::summarize(avg_air_temperature = mean(air_temperature)) %>%
  dplyr::arrange(site_id, day) %>%
  dplyr::group_by(site_id) %>%
  dplyr::mutate(rolling_air_temperature = zoo::rollapply(
    avg_air_temperature, 7, FUN = mean, align = 'left', fill = NA))

site_temps_plot <- daily_averages %>%
  ggplot(aes(x = day, y = rolling_air_temperature, color = site_id, group = site_id)) +
  geom_line() +
  theme_bw()

site_temps_plot
