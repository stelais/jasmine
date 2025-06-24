import RTModel
import shutil
import glob


def main(general_path,event_name,nprocessors,satellitedir):
    event_path = f'{general_path}/{event_name}'

    rtm = RTModel.RTModel()
    rtm.set_processors(nprocessors=nprocessors)
    rtm.set_event(event_path)
    rtm.archive_run()

    modeltypes = ['PS', 'LS', 'LX', 'LO']
    rtm.set_satellite_dir(satellitedir=satellitedir)
    peak_threshold = 5
    rtm.set_processors(nprocessors=nprocessors)
    rtm.set_event(event_path)
    rtm.set_satellite_dir(satellitedir=satellitedir)
    rtm.config_Reader(otherseasons=0, binning=1000000)
    rtm.config_InitCond(usesatellite=1, peakthreshold=peak_threshold, modelcategories=modeltypes)
    rtm.Reader()
    rtm.InitCond()

    # copy LX initial conditions from archived run
    shutil.copy(f'{event_path}/run-0001/InitCond/InitCondLX.txt', f'{event_path}/InitCond')
    # run fits on LX and LO
    rtm.launch_fits('LX')
    rtm.ModelSelector('LX')
    rtm.launch_fits('LO')
    rtm.ModelSelector('LO')

    #Copy PS and LS results to Models directory
    #I think this is really all we need for the finalizer to work, but we should do a detailed reading of the RTModel code
    #I did a semi-detailed reading only :)
    PS_Models = glob.glob(f'{event_path}/run-0001/Models/PS*')
    for model in PS_Models:
        shutil.copy(model, f'{event_path}/Models/')

    LS_Models = glob.glob(f'{event_path}/run-0001/Models/LS*')
    for model in LS_Models:
        shutil.copy(model, f'{event_path}/Models/')

    rtm.modelcodes = modeltypes #tell RTModel to only look for PS + binary models

    rtm.Finalizer() # Done

if __name__ == '__main__':
    general_path = '/Users/jmbrashe/VBBOrbital/NEWGULLS/MCMC/rerun_plx_OM'
    event_name = 'event_0_128_2350'
    satellitedir = '/Users/jmbrashe/VBBOrbital/NEWGULLS/MCMC/satellitedir'
    nprocessors = 5
    main(general_path=general_path,event_name=event_name,nprocessors=nprocessors,satellitedir=satellitedir)
