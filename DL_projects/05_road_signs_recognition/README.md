# Road Signs recognition with CNN #

## Problem ##
In this project I wanted to build a model that can recognize one of 30 different classes of road signs that are currently (July 2022) used in Kazakhstan. Actually, there are more than 250 different classes, but building such a massive model would require a lot of powerful hardware.

## Dataset ##
I could not find a prepared dataset in the Internet of Kazakhstani road signs, that is why I had to web scrape the official website and download road sign images from there. I also uploaded that project in my github repository: https://github.com/AlikhanNurkamal/web_scraping/tree/main/extracting_road_signs.

### Labels description ###
In the implementation stage the numeric representations of labels were used, so it might be not clear.
**Thus, here is the description of labels:**
* 0: Железнодорожный переезд со шлагбаумом
* 1: Железнодорожный переезд без шлагбаума
* 2: Однопутная железная дорога
* 3: Многопутная железная дорога
* 4-9: Приближение к железнодорожному переезду
* 10: Пересечение с трамвайной линией
* 11: Пересечение равнозначных дорог
* 12: Пересечение с круговым движением
* 13: Светофорное регулирование
* 14: Разводной мост
* 15: Выезд на набережную
* 16: Опасный поворот направо
* 17: Опасный поворот налево
* 18: Опасные повороты
* 19: Крутой спуск
* 20: Крутой подъем
* 21: Скользкая дорога
* 22: Неровная дорога
* 23: Искусственная неровность
* 24: Выброс гравия
* 25-27: Сужение дороги
* 28: Двустороннее движение
* 29: Пешеходный переход
* 30: Дети

## Results ##
As seen in the notebook, the `training loss` was 0.1719, `cross-validation loss` comprised 0.0653, and the `test set loss` was 0.0773. When it comes to the accuracy of the model, then the `training accuracy` is 94.06%, `validation accuracy` is 98.13%, and the `testing accuracy` is 97.75%. This implies that the model is better at generalizing to new, unseen data, rather than at the training process.

## Final thoughts ##
Well, the model does really poorly on the unseen data downloaded from the Internet (see last few rows in the jupyter notebook). So I cannot say that its testing accuracy is almost 98%. This might be due to the lack of different types of images of road signs, as all the images used in training are the generated versions of the same picture. This is why the model could be biased on only the provided images. In order to improve the model, I need a larger dataset with multiple images of the same road sign in different scenarios (one photo taken from the side, another taken right in front, etc.).
