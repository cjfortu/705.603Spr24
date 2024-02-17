# carsfactors
A random forest regressor trained to predict the days until the sale of a used car after posting the ad.

To run the model and services, you can simply run all cells in *carfactors.ipynb*, or build a docker image from *Dockerfile* and run the container.

**docker instructions**
1) 

## Microservice Info
**Provides two microservices**
1) returns performance stats - http://localhost:8786/stats
2) returns inference determination given vehicle and advertisement attributes - http://localhost:8786/infer?manufacturer=Subaru&transmission=automatic&color=blue&odometer=12000&year=2020&engine_type=gasoline&engine_capacity=3.6&bodytype=suv&warranty=True&drivetrain=all&price=20000&numphotos=10

**Details on microservices**
URL formation for inference/determination:  
*http[]()://localhost:8786/infer?manufacturer=str&transmission=str&color=str&odometer=int&year=int&engine_type=int&engine_capacity=float&bodytype=str&warranty=bool&drivetrain=str&price=int&numphotos=int*

URL formation to see model statistics:  
*http[]()://localhost:8786/stats*

manufacturer options:  
['Subaru', 'LADA', 'Dodge', 'УАЗ', 'Kia', 'Opel', 'Москвич', 'Alfa Romeo', 'Acura', 'Dacia', 'Lexus', 'Mitsubishi', 'Lancia', 'Citroen', 'Mini', 'Jaguar', 'Porsche', 'SsangYong', 'Daewoo', 'Geely', 'ВАЗ', 'Fiat', 'Ford', 'Renault', 'Seat', 'Rover', 'Volkswagen', 'Lifan', 'Jeep', 'Cadillac', 'Audi', 'ЗАЗ', 'Toyota', 'ГАЗ', 'Volvo', 'Chevrolet', 'Great Wall', 'Buick', 'Pontiac', 'Lincoln', 'Hyundai', 'Nissan', 'Suzuki', 'BMW', 'Mazda', 'Land Rover', 'Iveco', 'Skoda', 'Saab', 'Infiniti', 'Chery', 'Honda', 'Mercedes-Benz', 'Peugeot', 'Chrysler']

transmission options:  
['automatic', 'mechanical']

color options:  
['silver', 'blue', 'red', 'black', 'grey', 'other', 'brown', 'white', 'green', 'violet', 'orange', 'yellow']

engine_type options:  
['gasoline', 'diesel', 'electric']

body_type options:  
['universal', 'suv', 'sedan', 'hatchback', 'liftback', 'minivan', 'minibus', 'van', 'pickup', 'coupe', 'cabriolet', 'limousine']

warranty options:  
True, False

drivetrain options:  
['all', 'front', 'rear']

## Model Performance
The model has an R2 score of 0.616, which means %61.6 of the variance in the data is captured by the model.  This is considered good.  It is possible to achieve an R2 score as high as 0.85, but this results in an unacceptably overfitted model.

The model has an RMSE of 70.305 in training and 83.330 in testing.  These are both within the label standard deviation of 112.837.  Again we can achieve an RMSE as low as 42 in training, but this results in an unacceptably overfitted model.

Employing a fully connected neural network (FCN) could capture the nonlinear relationships between inputs and outputs.  This may require an architecture prohibitively large for CPU training (CPU training is necessarily the case using Sklearn).  If done in Tensorflow using a GPU, we could use 3 hidden layers with many neurons, and employ variance reduction in the form of dropout.