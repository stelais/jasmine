from setuptools import setup

setup(name='jasmine-astro',
      version='0.1.4.6',
      description='JASMINE: Joint Analysis of Simulations for Microlensing INterest Events',
      url='https://github.com/stelais/jasmine',
      author='Stela IS, Jon B., C.R., JASMINE authors',
      author_email='stela.ishitanisilva@nasa.gov',
      license='MIT',
      packages=['jasmine',
                'jasmine.classes_and_files_reader',
                'jasmine.files_organizer',
                'jasmine.investigator',
                'jasmine.lightcurve_signal_generator'],
      platforms="Mac OS 14.5",
      keywords=['Astronomy', 'Microlensing', 'Science'],
      zip_safe=False,
      install_requires=["pandas==2.1.4",
                        "jupyter==1.0.0",
                        "notebook==7.1.3",
                        "ipykernel==6.29.4",
                        "matplotlib==3.8.4",
                        "astropy==6.1.0",
                        "jplephem==2.22",
                        "bokeh==3.4.1",
                        "VBBinaryLensing==3.7.0",
                        "moana-pypi==0.2.2"
                        ])
