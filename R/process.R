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
  dplyr::mutate(day = as.integer(format(timestamp, "%d")),
                week = as.integer(format(timestamp, "%U")))

buildings_per_site <- buildings %>%
  dplyr::group_by(site_id) %>%
  dplyr::summarize(dplyr::n())
buildings_per_site

## No, this isn't weekly! Rows in weather are by the minute, not the day.
#weather %<>% 
#  dplyr::arrange(site_id, timestamp) %>%
#  dplyr::group_by(site_id) %>%
#  dplyr::mutate(weekly_avg_air_temperature = zoo::rollapply(
#    air_temperature, 7, mean, align='left', fill=NA))
  
site_temps_plot <- weather %>%
  ggplot(aes(x = timestamp, y = weekly_avg_air_temperature)) +
  geom_line() +
  theme_bw() +
  facet_wrap(~site_id)
site_temps_plot
