import matplotlib.pyplot as plt

emissions = [44234235.68, 43229906.19, 44025860.01, 45313335.21, 45584445.82,
             43671986.90, 45446390.30, 46900918.42, 42931460.89, 45514464.62, 50270225.23
             ]

plt.figure()
x = list(range(2012, 2023))

plt.plot(x, emissions, '#F6B008', marker='o')

plt.xlabel('year')
plt.ylabel('Global carbon emissions(Kt)')

plt.title('Trends in global carbon emissions from 2012 to 2022')

plt.show()


