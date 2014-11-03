#include "vibe2.h"
int N=20;//knob
int R=20;//knob

#define rndSize 256
unsigned char ri=0;
#define rdx ri++

int noMin=2;//knob
int phi=16;//knob//ghost?//int phi=0;//dint work?pseudo code.oops a case of premature d..bitching
RNG rnd = theRNG();

int rndp[rndSize],rndn[rndSize],rnd8[rndSize];
struct model {
    Mat*** samples;
    Mat** fgch;
    Mat* fg;
};
vector<model*> models;
int initDone;

uchar table[256];
Mat lookUpTable(1, 256, CV_8U);
void init_vibe()
{
    for (int i = 0; i < 256; ++i)
        table[i] = (uchar)(20 * (i/20));

    uchar* p = lookUpTable.data;
    for( int i = 0; i < 256; ++i)
        p[i] = table[i];

    for(int i=0;i<rndSize;i++)
    {
        rndp[i]=rnd(phi);
        rndn[i]=rnd(N);
        rnd8[i]=rnd(8);
    }
}

int init_model(Mat& firstSample)
{
    vector<Mat> channels;
    split(firstSample,channels);
    if(!initDone)
    {
        init_vibe();
        initDone=0;
    }
    model* m=new model;
    m->fgch= new Mat*[channels.size()];
    m->samples=new Mat**[N];
    m->fg=new Mat(Size(firstSample.cols,firstSample.rows), CV_8UC1);
    for(int s=0;s<channels.size();s++)
    {
        m->fgch[s]=new Mat(Size(firstSample.cols,firstSample.rows), CV_8UC1);
        Mat** samples= new Mat*[N];
        for(int i=0;i<N;i++)
        {
            samples[i]= new Mat(Size(firstSample.cols,firstSample.rows), CV_8UC1);
        }
        for(int i=0;i<channels[s].rows;i++)
        {
            int ioff=channels[s].step.p[0]*i;
            for(int j=0;j<channels[0].cols;j++)
            {
                for(int k=0;k<N;k++)
                {
                    (samples[k]->data + ioff)[j]=channels[s].at<uchar>(i,j);
                }
                (m->fgch[s]->data + ioff)[j]=0;

                if(s==0)(m->fg->data + ioff)[j]=0;
            }
        }
        m->samples[s]=samples;
    }
    models.push_back(m);
    return models.size()-1;
}

void fg_vibe1Ch(Mat& frame,Mat** samples,Mat* fg)
{
    int step=frame.step.p[0];
    //#pragma omp parallel for
    for(int i=1;i<frame.rows-1;i++)
    {
        int ioff= step*i;
        //#pragma omp parallel for
        for(int j=1;j<frame.cols-1;j++)
        {
            int count =0,index=0;
            while((count<noMin) && (index<N))
            {
            //hotspot but may need to untangle the waves somewhere else
                int dist= (samples[index]->data + ioff)[j]-(frame.data + ioff)[j];
                if(dist<=R && dist>=-R)
                {
                    count++;
                }
                index++;
            }
            if(count>=noMin)
            {
                ((fg->data + ioff))[j]=0;
                int rand= rndp[rdx];
                if(rand==0)
                {
                    rand= rndn[rdx];
                    (samples[rand]->data + ioff)[j]=(frame.data + ioff)[j];
                }
                rand= rndp[rdx];
                int nxoff=ioff;
                if(rand==0)
                {
                    int nx=i,ny=j;
                    int cases= rnd8[rdx];
                    switch(cases)
                    {
                    case 0:
                        //nx--;
                        nxoff=ioff-step;
                        ny--;
                        break;
                    case 1:
                        //nx--;
                        nxoff=ioff-step;
                        ny;
                        break;
                    case 2:
                        //nx--;
                        nxoff=ioff-step;
                        ny++;
                        break;
                    case 3:
                        //nx++;
                        nxoff=ioff+step;
                        ny--;
                        break;
                    case 4:
                        //nx++;
                        nxoff=ioff+step;
                        ny;
                        break;
                    case 5:
                        //nx++;
                        nxoff=ioff+step;
                        ny++;
                        break;
                    case 6:
                        //nx;
                        ny--;
                        break;
                    case 7:
                        //nx;
                        ny++;
                        break;
                    }
                    rand= rndn[rdx];
                    (samples[rand]->data + nxoff)[ny]=(frame.data + ioff)[j];
                }
            }else
            {
                ((fg->data + ioff))[j]=255;
            }
        }
    }
}

Mat* fg_vibe(Mat& frame,int idx)
{
    vector<Mat> channels;
    split(frame,channels);
    //#pragma omp parallel for
    for(int i=0;i<channels.size();i++)
    {
        LUT(channels[i], lookUpTable, channels[i]);
        fg_vibe1Ch(channels[i],models[idx]->samples[i],models[idx]->fgch[i]);
        if(i>0 && i<2)
        {
            bitwise_or(*models[idx]->fgch[i-1],*models[idx]->fgch[i],*models[idx]->fg);
        }
        if(i>=2)
        {
            bitwise_or(*models[idx]->fg,*models[idx]->fgch[i],*models[idx]->fg);
        }
    }
    if(channels.size()==1) return models[idx]->fgch[0];
    return models[idx]->fg;
}
