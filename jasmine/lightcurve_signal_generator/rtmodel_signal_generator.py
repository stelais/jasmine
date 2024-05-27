from jasmine import RTModelTemplateForBinaryLightCurve


def easy_plot_lightcurve(rtmodel_template_two_lenses_):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    magnification, time_interval = rtmodel_template_two_lenses_.rtmodel_magnification_using_vbb()
    ax.scatter(time_interval, magnification, s=0.8, alpha=0.3)
    plt.show(dpi=600)


if __name__ == '__main__':
    template_path = '/Users/stela/Documents/Scripts/RTModel_project/RTModel/RTModel/data/TemplateLibrary.txt'
    rtmodel_template_two_lenses = RTModelTemplateForBinaryLightCurve(template_line=2,
                                                                     path_to_template=template_path,
                                                                     input_peak_t1=300,
                                                                     input_peak_t2=302)
    easy_plot_lightcurve(rtmodel_template_two_lenses)

