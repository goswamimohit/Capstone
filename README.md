![c059e222-21d0-40af-8a5a-c24d061bb185](https://user-images.githubusercontent.com/95187592/175029508-ec4ffa41-a650-4592-bea3-1144574d3eb1.png)


Olympic Historical DataSet
--------------------------------------------------------
--------------------------------------------------------

## Framework and Language Used in this Project
- Python
- SQL
- PySpark
- Pandas

## Introduction
The Olympics are regarded as a major athletic event in which thousands of competitors from all over the world compete in a range of events. Nations from all over the world compete, and the Olympic games are often regarded as the world's most popular sporting event. Data Science and Machine Learning techniques will be of significant assistance in the decision-making processes of trainers, players, and governments in these countries.

The findings may be used to highlight the need for new policies to increase the quality of physical education in a country. According to the data, a variety of factors contribute to these countries' performance in the games.

The following dataset contains information about olympic events that occurred between 2000 and 2012. Summer and winter olympic sports are also included. The dataset has a total of ten features. 
                        There are 8618 observations.

                        Players' names, ages, countries, medal types, and total medals earned are all included in the data collection.
                        
                        The dataset also assume Madel won in group sports also as individual Medal for each player.

                        A single player can play many games.

                        Year - The year in which a certain olympic event took place.

                        Date Given - The date on which a sporting event will take place.

                        Same A player can compete in a number of Olympic events.

## Follwowing Steps Are Performed in this Project
1. Data Cleaning / Preprocessing - Pandas

        - Deal with missing values appropriately. You can either remove them or fill them, but a proper justification is required.
     
      ![c4f0ed626e109ed02f30bdeeb85b38881451f252](https://user-images.githubusercontent.com/95187592/175030297-a2db8203-d368-4e61-954a-20d046c6eb3b.png)

        
        - Duplicates in the dataset introduces bias in the study. Please check and perform appropriate steps.
        
        - Please remove special characters from the name column.
        
3. Inserting Data in SQL from CSV file  - PyMySQl Library
4. Normalize Database
5. Entity Relationship Diagram from Normalize Database

     ![Olympic ER Diagram Normalize](https://user-images.githubusercontent.com/95187592/175028768-aa081c10-8a37-4aa6-bec2-0d4cdb7f51a9.png)  

7. Analyzing Data with help of SQL Query
         - Find the average number of medals won by each country
         
         - Display the countries and the number of gold medals they have won in decreasing order
         
         - Display the list of people and the medals they have won in descending order, grouped by their country
         
         - Display the list of people with the medals they have won according to their their age
         
         - Which country has won the most number of medals (cumulative)
        
7. Analyzing Data with help of Pyspark Framewoek

        - Write PySpark code to print the Olympic Sports/games in the dataset.
        
        - Write PySpark code to plot the total number of medals in  each Olympic Sport/game
       
        - Sort the result based on the total number of medals.
        
        - Find the total number of medals won by each country in swimming.
        
        - Find the total number of medals won by each country in Skeleton.
       
        - Find the number of medals that the US won yearly.
       
        - Find the total number of medals won by each country.
       
        - Who was the oldest athlete in the olympics? 
       
        - Which country was he/she from?
      
9. Exploratory Data Analysis and Visualization with Pandas

      ![2b26d31fe2180048994a976f1f2ff3d5519be2a8](https://user-images.githubusercontent.com/95187592/175029656-9a682612-1725-4dc0-9867-b5ed8d27b198.png)
      ![377d557892d62e47680e5df7786f0965bf63d3d9](https://user-images.githubusercontent.com/95187592/175029728-ec2e3f0e-8693-4de5-9b4a-37a189e235f9.png)
      ![6b4dcb997c16c4b417e52c9217c6c2a42fcad085](https://user-images.githubusercontent.com/95187592/175029797-70d24d17-15ec-4226-9a07-bb09ab5342a9.png)
      ![8beeb6c509b04e5d7e7b20ef0678384c5fe64b69](https://user-images.githubusercontent.com/95187592/175029842-786828f6-25be-42e4-8593-9c476bb0991d.png)
      ![9985cca057871aabbde164088a927865d784c640](https://user-images.githubusercontent.com/95187592/175029883-48f14196-1992-4049-9fcd-2b957aae5deb.png)
      ![132024e52e7c8f4786cab0834d87a8ed6eb3b053](https://user-images.githubusercontent.com/95187592/175029930-cd827cce-7a49-4770-b844-9c44c21c03a8.png)

 
11. Feature Enginnering in above of this steps.

          - Please check if the age dataset is skewed or symmetric. Based on results perform transformation.
          
       ![3e976cd42acea61a1ccd05c3611e15a4cbc9a50d](https://user-images.githubusercontent.com/95187592/175029131-073c50f6-3688-49fc-9437-71598a787e11.png)



