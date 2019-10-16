library(glue)
library(ggplot2)

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
)

buildings <- readr::read_csv(files$buildings)
meter <- readr::read_csv(files$meter$train)
weather <- files$weather(files$weather$train)
