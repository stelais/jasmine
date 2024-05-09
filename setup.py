from setuptools import setup

setup(name='jasmine-sis',
      version='0.1.0.0.7',
      description='JASMINE: Joint Analysis of Simulations for Microlensing INterest Events',
      url='https://github.com/stelais/jasmine',
      author='Stela IS',
      author_email='stela.ishitanisilva@nasa.gov',
      license='MIT',
      packages=['jasmine',
                'jasmine.files_organizer'],
      zip_safe=False,
      install_requires=["pandas==2.1.4",
                        "jupyter==1.0.0",
                        "notebook==7.1.3",
                        "ipykernel==6.29.4",
                        "matplotlib==3.8.4",
                        "astropy==6.1.0",
                        "jplephem==2.22"
                        ])
