# Load in the keras package
#install.packages(keras)
library(keras)

# Install TensorFlow
#install.packages(tensorflow)
library(tensorflow)

# Install xlsx package to read xls/xlsx file
#install.packages(httr)
library(httr)
#install.packages(RCurl)
library(RCurl)
#install.packages(readxl)
library(readxl)



main <- function() 
{
  choice <- function(){
    choice <- readline("Choose option 1 for Boston dataset, option 2 for Default dataset (1/2): ")
    choice <- as.integer(choice)
    return(choice)
  }
  choice()
  
  if (choice() == 1) {
    install.packages(MASS) # Import the MASS library
    library(MASS)
    dataset <- attach(Boston) # Import the Boston dataset
  }
  
  else if (choice() == 2) { 
    url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    GET(url, write_disk("default.xls", overwrite=TRUE))
    dataset <- read_xls('default.xls', sheet = 1, skip = 1)
  } 
  
  else {
    break
  }
}

if(!interactive()) {
  main()
}
