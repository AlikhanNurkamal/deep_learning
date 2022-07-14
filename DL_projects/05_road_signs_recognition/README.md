# Road Signs recognition with CNN #

## Problem ##
In this project I wanted to build a model that can recognize one of 30 different classes of road signs that are currently (July 2022) used in Kazakhstan. Actually, there are more than 250 different classes, but building such a massive model would require a lot of powerful hardware.

## Dataset ##
I could not find a prepared dataset in the Internet of Kazakhstani road signs, that is why I had to web scrape the official website and download road sign images from there. I also uploaded that project in my github repository: https://github.com/AlikhanNurkamal/web_scraping/tree/main/extracting_road_signs.

### Labels description ###
In the implementation stage the numeric representations of labels were used, so it might be not clear.
** Thus, here is the description of labels: **
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
