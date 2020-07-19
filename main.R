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
  choice <- function()
  {
    # ask user for their choice
    choice <- readline(prompt="Choose option 1 for Boston dataset, option 2 for Default dataset, option 3 to quit (1/2/3): ")
    # convert character into integer
    choice <- as.integer(choice)
    return(choice)
  }
  
  while (choice != 3) 
  {
    if (choice == 1) # Boston dataset
    {
      # Import the MASS library
      install.packages(MASS)
      library(MASS)
      
      # Import the Boston dataset
      dataset <- attach(Boston)
    }
    
    else if (choice == 2) # Default dataset
    {
      url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
      GET(url, write_disk("default.xls", overwrite=TRUE))
      dataset <- read_xls('default.xls', sheet = 1, skip = 1)
    } 
    
    else # Wrong choice
    {
      print("Wrong choice. Please try again.")
    }
  }
   
}

if(!interactive()) {
  main()
}
